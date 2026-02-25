"""FastAPI backend for Speaker Separation web app.

Run with:  python server.py
           or: uvicorn server:app --reload --port 8000

Pipelines:
  - energy:    Local energy-based (instantaneous, no GPU)
  - channel:   pyannote 3.1 + Channel-Aware Wiener (Modal GPU)
  - sepformer: pyannote 3.1 + SepFormer neural separation (Modal GPU)
  - nemo:      NVIDIA NeMo MSDD + TitaNet (Modal GPU)
  - diarizen:  DiariZen (VoxSRC 2023 winner) + CAM++ (Modal GPU)

Endpoints:
  POST /api/process/{session_num}           — launch all 5 pipelines
  GET  /api/process/{job_id}/status         — SSE job progress stream
  GET  /api/results/{session_num}           — pipeline comparison results
  GET  /api/results/{session_num}/{pipe}/{track} — serve audio file
  POST /api/transcribe/{session_num}        — Whisper transcription (SSE)
  GET  /api/transcript/{session_num}        — get stored transcript
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import MODAL_APP_NAME, VOLUME_THRESHOLD_BYTES
from src.audio_utils import (
    discover_sessions,
    find_session_folder,
    load_session_audio_separate,
    save_wav_bytes,
)
from src.energy_pipeline import run_energy_pipeline

AUDIO_DIR = Path("Audio_Files")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TRACKS = ["speaker_a", "speaker_b"]

# All pipelines: energy runs locally, the rest on Modal GPU
PIPELINES = ["energy", "channel", "sepformer", "nemo", "diarizen"]
PIPELINE_LABELS = {
    "energy":    "Energy-Based",
    "channel":   "Pyannote + Channel Wiener",
    "sepformer": "Pyannote + SepFormer",
    "nemo":      "NVIDIA NeMo MSDD",
    "diarizen":  "DiariZen (VoxSRC)",
}
PIPELINE_DESCRIPTIONS = {
    "energy":    "Fast heuristic, no GPU. Channel energy assignment.",
    "channel":   "Pyannote 3.1 diarization + leakage suppression + Wiener overlap separation.",
    "sepformer": "Pyannote 3.1 + SpeechBrain SepFormer neural source separation for overlaps.",
    "nemo":      "NVIDIA NeMo MSDD multi-scale diarizer + TitaNet speaker embeddings.",
    "diarizen":  "DiariZen (VoxSRC 2023 winner) end-to-end + CAM++ speaker embeddings.",
}

app = FastAPI(title="Speaker Separation", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job & transcript store
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}
transcripts: dict[str, dict] = {}    # session_name -> transcript data


def _get_pipeline():
    Pipeline = modal.Cls.from_name(MODAL_APP_NAME, "SpeakerSeparationPipeline")
    return Pipeline()


def _session_output_dir(session_name: str, pipeline_id: str) -> Path:
    return OUTPUT_DIR / session_name / pipeline_id


def _session_has_results(session_name: str) -> bool:
    """Check if any pipeline has results for this session."""
    for pid in PIPELINES:
        d = _session_output_dir(session_name, pid)
        if d.exists() and all((d / f"{t}.wav").exists() for t in TRACKS):
            return True
    return False


def _transcript_path(session_name: str) -> Path:
    return OUTPUT_DIR / session_name / "transcript.json"


# ---------------------------------------------------------------------------
# API: Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    try:
        modal.Cls.from_name(MODAL_APP_NAME, "SpeakerSeparationPipeline")
        return {"status": "ok", "detail": "Modal app deployed and reachable", "pipelines": PIPELINES}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# API: Sessions
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
async def list_sessions():
    if not AUDIO_DIR.exists():
        return {"sessions": []}

    session_nums = discover_sessions(AUDIO_DIR)
    sessions = []

    for num in session_nums:
        folder = find_session_folder(AUDIO_DIR, num)
        if folder is None:
            continue

        wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
        seen = set()
        unique = []
        for f in wav_files:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                unique.append(f)

        total_bytes = sum(f.stat().st_size for f in unique)
        session_name = f"session_{num:03d}"
        has_results = _session_has_results(session_name)
        has_transcript = _transcript_path(session_name).exists()

        active_job = None
        for jid, j in jobs.items():
            if j["session_num"] == num and j["status"] in ("pending", "processing"):
                active_job = jid
                break

        sessions.append({
            "num": num,
            "folder": folder.name,
            "files": [f.name for f in unique],
            "num_files": len(unique),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 1),
            "has_results": has_results,
            "has_transcript": has_transcript,
            "active_job": active_job,
        })

    return {"sessions": sessions}


# ---------------------------------------------------------------------------
# API: Upload
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_files(
    session_name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    session_dir = AUDIO_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = session_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append(f.filename)

    return {"session": session_name, "files_saved": saved}


# ---------------------------------------------------------------------------
# API: Process (runs all 5 pipelines)
# ---------------------------------------------------------------------------


def _save_pipeline_results(session_name: str, pipeline_id: str, results: dict):
    """Save speaker_a.wav and speaker_b.wav into output/{session}/{pipeline}/."""
    out_dir = _session_output_dir(session_name, pipeline_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        if track in results and results[track]:
            save_wav_bytes(results[track], out_dir / f"{track}.wav")


async def _run_processing(job_id: str, session_num: int):
    job = jobs[job_id]
    job["status"] = "processing"
    job["started_at"] = time.time()

    try:
        folder = find_session_folder(AUDIO_DIR, session_num)
        if folder is None:
            raise FileNotFoundError(f"Session {session_num} folder not found")

        loop = asyncio.get_event_loop()
        channels, info = await loop.run_in_executor(
            None, load_session_audio_separate, folder
        )

        job["stage"] = "loaded"
        job["audio_info"] = {
            "num_files": info["num_files"],
            "duration_s": round(info["duration_s"], 1),
            "size_mb": round(info["total_bytes"] / (1024 * 1024), 1),
        }

        session_name = f"session_{session_num:03d}"

        # --- Pipeline A: Energy-based (local, near-instant) ---
        job["stage"] = "energy_processing"
        t0 = time.time()
        energy_results = await loop.run_in_executor(
            None,
            run_energy_pipeline,
            channels["channel_a_bytes"],
            channels["channel_b_bytes"],
        )
        energy_elapsed = round(time.time() - t0, 1)
        _save_pipeline_results(session_name, "energy", energy_results)
        job["stage"] = "energy_done"
        job["pipeline_metadata"] = {
            "energy": {
                "elapsed_s": energy_elapsed,
                "segments": 0,
                "overlap_pct": 0,
                "model": "energy",
            }
        }

        # --- Pipelines B, C, D, E: ML-based (Modal GPU) ---
        job["stage"] = "gpu_processing"
        pipeline = _get_pipeline()

        # Use process_all_pipelines if deployed; fall back to process_session_compare
        # for older deployments that don't have NeMo/DiariZen yet.
        def _call_gpu(pipe, ch_a, ch_b, mixed, sname):
            try:
                return pipe.process_all_pipelines.remote(ch_a, ch_b, mixed, sname)
            except Exception as e:
                if "process_all_pipelines" in str(e) or "has no method" in str(e).lower():
                    print(f"[WARN] process_all_pipelines not found on deployed Modal class, "
                          f"falling back to process_session_compare: {e}")
                    old = pipe.process_session_compare.remote(ch_a, ch_b, mixed, sname)
                    return old  # returns {"channel": ..., "sepformer": ...}
                raise

        gpu_results = await loop.run_in_executor(
            None,
            _call_gpu,
            pipeline,
            channels["channel_a_bytes"],
            channels["channel_b_bytes"],
            channels["mixed_bytes"],
            session_name,
        )

        # Save each ML pipeline result
        pipeline_meta = {"energy": job["pipeline_metadata"]["energy"]}

        for pid in ["channel", "sepformer", "nemo", "diarizen"]:
            if pid in gpu_results:
                result = gpu_results[pid]
                _save_pipeline_results(session_name, pid, result)
                meta = result.get("metadata", {})
                pipeline_meta[pid] = {
                    "elapsed_s": meta.get("elapsed_s", 0),
                    "segments": meta.get("segments", 0),
                    "overlap_pct": meta.get("overlap_pct", 0),
                    "model": meta.get("model", pid),
                    "error": meta.get("error"),
                }

        job["pipeline_metadata"] = pipeline_meta
        job["stage"] = "saving"

        # --- Auto-transcribe with the best pipeline's diarization ---
        job["stage"] = "transcribing"
        try:
            transcript_result = await loop.run_in_executor(
                None,
                pipeline.transcribe_session.remote,
                channels["channel_a_bytes"],
                channels["channel_b_bytes"],
                channels["mixed_bytes"],
                session_name,
                None,  # language = auto
            )
            _transcript_path(session_name).parent.mkdir(parents=True, exist_ok=True)
            _transcript_path(session_name).write_text(json.dumps(transcript_result))
            transcripts[session_name] = transcript_result
        except Exception as te:
            print(f"[WARN] Transcription failed: {te}")

        job["status"] = "done"
        job["stage"] = "complete"
        job["finished_at"] = time.time()
        job["elapsed_s"] = round(job["finished_at"] - job["started_at"], 1)

    except Exception as e:
        job["status"] = "error"
        job["stage"] = "failed"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        if job.get("started_at"):
            job["elapsed_s"] = round(job["finished_at"] - job["started_at"], 1)


@app.post("/api/process/{session_num}")
async def start_processing(session_num: int, force: bool = False):
    """Launch all pipelines for a session.

    ?force=true  — re-run even if results already exist (reprocess / redeploy results).
    """
    # Block duplicate jobs unless forced
    for jid, j in jobs.items():
        if j["session_num"] == session_num and j["status"] in ("pending", "processing"):
            return {"job_id": jid, "status": j["status"], "message": "Already processing"}

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "session_num": session_num,
        "status": "pending",
        "stage": "queued",
        "created_at": time.time(),
        "pipeline_metadata": {},
        "forced": force,
    }

    asyncio.create_task(_run_processing(job_id, session_num))
    return {"job_id": job_id, "status": "pending", "forced": force}


@app.get("/api/process/{job_id}/status")
async def job_status_sse(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        while True:
            job = jobs.get(job_id)
            if job is None:
                break

            data = json.dumps(job, default=str)
            yield f"data: {data}\n\n"

            if job["status"] in ("done", "error"):
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# API: Results (per-pipeline)
# ---------------------------------------------------------------------------


@app.get("/api/results/{session_num}")
async def get_results(session_num: int):
    session_name = f"session_{session_num:03d}"
    pipelines = {}

    for pid in PIPELINES:
        d = _session_output_dir(session_name, pid)
        tracks = {}
        for track in TRACKS:
            path = d / f"{track}.wav"
            if path.exists():
                tracks[track] = {
                    "filename": path.name,
                    "size_kb": round(path.stat().st_size / 1024),
                    "url": f"/api/results/{session_num}/{pid}/{track}",
                }

        if tracks:
            meta = {"elapsed_s": 0, "segments": 0, "overlap_pct": 0, "model": pid}
            for j in jobs.values():
                if j.get("session_num") == session_num and j.get("pipeline_metadata"):
                    pm = j["pipeline_metadata"].get(pid)
                    if pm:
                        meta = pm
                    break

            pipelines[pid] = {
                "id": pid,
                "label": PIPELINE_LABELS.get(pid, pid),
                "description": PIPELINE_DESCRIPTIONS.get(pid, ""),
                "tracks": tracks,
                "elapsed_s": meta.get("elapsed_s", 0),
                "segments": meta.get("segments", 0),
                "overlap_pct": meta.get("overlap_pct", 0),
                "model": meta.get("model", pid),
                "error": meta.get("error"),
            }

    if not pipelines:
        raise HTTPException(status_code=404, detail="No results found")

    return {"session_num": session_num, "pipelines": pipelines}


@app.get("/api/results/{session_num}/{pipeline}/{track}")
async def serve_track(session_num: int, pipeline: str, track: str):
    if pipeline not in PIPELINES:
        raise HTTPException(status_code=400, detail="Invalid pipeline")
    if track not in TRACKS:
        raise HTTPException(status_code=400, detail="Invalid track name")

    session_name = f"session_{session_num:03d}"
    path = _session_output_dir(session_name, pipeline) / f"{track}.wav"

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type="audio/wav", filename=f"{session_name}_{pipeline}_{track}.wav")


# ---------------------------------------------------------------------------
# API: Transcription
# ---------------------------------------------------------------------------


@app.post("/api/transcribe/{session_num}")
async def start_transcription(session_num: int):
    """Trigger Whisper transcription for a session (already processed)."""
    session_name = f"session_{session_num:03d}"
    folder = find_session_folder(AUDIO_DIR, session_num)
    if folder is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if transcript already exists
    tp = _transcript_path(session_name)
    if tp.exists():
        data = json.loads(tp.read_text())
        return {"status": "cached", "session_num": session_num, "transcript": data}

    # Run transcription async
    job_id = f"transcript_{session_num}"

    async def _do_transcribe():
        loop = asyncio.get_event_loop()
        try:
            channels, _ = await loop.run_in_executor(
                None, load_session_audio_separate, folder
            )
            pipeline = _get_pipeline()
            result = await loop.run_in_executor(
                None,
                pipeline.transcribe_session.remote,
                channels["channel_a_bytes"],
                channels["channel_b_bytes"],
                channels["mixed_bytes"],
                session_name,
                None,
            )
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(json.dumps(result))
            transcripts[session_name] = result
        except Exception as e:
            transcripts[session_name] = {"error": str(e)}

    asyncio.create_task(_do_transcribe())
    return {"status": "processing", "session_num": session_num}


@app.get("/api/transcript/{session_num}")
async def get_transcript(session_num: int):
    """Get stored transcript for a session."""
    session_name = f"session_{session_num:03d}"

    # Check in-memory cache first
    if session_name in transcripts:
        return {"session_num": session_num, **transcripts[session_name]}

    # Check disk
    tp = _transcript_path(session_name)
    if tp.exists():
        data = json.loads(tp.read_text())
        transcripts[session_name] = data
        return {"session_num": session_num, **data}

    raise HTTPException(status_code=404, detail="No transcript found")


# ---------------------------------------------------------------------------
# API: Pipeline metadata / comparison
# ---------------------------------------------------------------------------


@app.get("/api/pipelines")
async def list_pipelines():
    """Return metadata about all available pipelines."""
    return {
        "pipelines": [
            {
                "id": pid,
                "label": PIPELINE_LABELS[pid],
                "description": PIPELINE_DESCRIPTIONS[pid],
            }
            for pid in PIPELINES
        ]
    }


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = frontend_dist / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
