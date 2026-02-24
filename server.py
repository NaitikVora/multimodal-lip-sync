"""FastAPI backend for Speaker Separation web app.

Run with:  python server.py
           or: uvicorn server:app --reload --port 8000
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
PIPELINES = ["energy", "channel", "sepformer"]
PIPELINE_LABELS = {
    "energy": "Energy-Based",
    "channel": "ML + Channel-Aware",
    "sepformer": "ML + SepFormer",
}

app = FastAPI(title="Speaker Separation", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


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


# ---------------------------------------------------------------------------
# API: Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    try:
        modal.Cls.from_name(MODAL_APP_NAME, "SpeakerSeparationPipeline")
        return {"status": "ok", "detail": "Modal app deployed and reachable"}
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
# API: Process (runs all 3 pipelines)
# ---------------------------------------------------------------------------


def _save_pipeline_results(session_name: str, pipeline_id: str, results: dict):
    """Save speaker_a.wav and speaker_b.wav into output/{session}/{pipeline}/."""
    out_dir = _session_output_dir(session_name, pipeline_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        if track in results:
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

        # --- Pipelines B + C: ML-based (Modal GPU) ---
        job["stage"] = "gpu_processing"
        pipeline = _get_pipeline()

        gpu_results = await loop.run_in_executor(
            None,
            pipeline.process_session_compare.remote,
            channels["channel_a_bytes"],
            channels["channel_b_bytes"],
            channels["mixed_bytes"],
            session_name,
        )

        # Save Pipeline B (channel-aware)
        _save_pipeline_results(session_name, "channel", gpu_results["channel"])
        channel_meta = gpu_results["channel"].get("metadata", {})

        # Save Pipeline C (sepformer)
        _save_pipeline_results(session_name, "sepformer", gpu_results["sepformer"])
        sepformer_meta = gpu_results["sepformer"].get("metadata", {})

        job["stage"] = "saving"

        # Store per-pipeline metadata in the job
        job["pipeline_metadata"] = {
            "energy": {
                "elapsed_s": energy_elapsed,
                "segments": 0,
                "overlap_pct": 0,
            },
            "channel": {
                "elapsed_s": channel_meta.get("elapsed_s", 0),
                "segments": channel_meta.get("segments", 0),
                "overlap_pct": channel_meta.get("overlap_pct", 0),
            },
            "sepformer": {
                "elapsed_s": sepformer_meta.get("elapsed_s", 0),
                "segments": sepformer_meta.get("segments", 0),
                "overlap_pct": sepformer_meta.get("overlap_pct", 0),
            },
        }

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
async def start_processing(session_num: int):
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
    }

    asyncio.create_task(_run_processing(job_id, session_num))
    return {"job_id": job_id, "status": "pending"}


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
            # Try to find metadata from the most recent job
            meta = {"elapsed_s": 0, "segments": 0, "overlap_pct": 0}
            for j in jobs.values():
                if j.get("session_num") == session_num and j.get("pipeline_metadata"):
                    pm = j["pipeline_metadata"].get(pid)
                    if pm:
                        meta = pm
                    break

            pipelines[pid] = {
                "id": pid,
                "label": PIPELINE_LABELS[pid],
                "tracks": tracks,
                "elapsed_s": meta.get("elapsed_s", 0),
                "segments": meta.get("segments", 0),
                "overlap_pct": meta.get("overlap_pct", 0),
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
