"""NeMo-based speaker separation pipeline — runs INSIDE Modal GPU container.

Uses NVIDIA NeMo's ClusteringDiarizer (MSDD + TitaNet embeddings) for
speaker diarization, then applies the same channel-aware separation stages
from the main pipeline.

NeMo advantages over pyannote:
  - MSDD (Multi-Scale Diarization Decoder) handles overlapping speech better
  - TitaNet speaker embeddings trained on broader multilingual data
  - Superior performance on far-field / meeting room recordings
"""

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# NeMo Diarization
# ---------------------------------------------------------------------------


def diarize_nemo(
    audio: np.ndarray,
    sr: int,
    num_speakers: int = 2,
) -> list[dict]:
    """Run NeMo ClusteringDiarizer.

    Returns [{"speaker": str, "start": float, "end": float}, ...]
    """
    try:
        from omegaconf import OmegaConf          # noqa: F401 (presence check)
        from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
    except ImportError as e:
        print(f"[NeMo] Package not available ({e})")
        return []

    return _diarize_nemo_native(audio, sr, num_speakers)


def _diarize_nemo_native(
    audio: np.ndarray,
    sr: int,
    num_speakers: int = 2,
) -> list[dict]:
    """Full NeMo MSDD diarization (only called when NeMo is installed)."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_path = tmpdir / "audio.wav"

        # Write audio to temp WAV file
        sf.write(str(audio_path), audio, sr)

        # Create manifest file (NeMo requires a manifest)
        manifest_path = tmpdir / "manifest.json"
        import json
        manifest_entry = {
            "audio_filepath": str(audio_path),
            "offset": 0.0,
            "duration": len(audio) / sr,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(json.dumps(manifest_entry) + "\n")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # NeMo diarization config
        cfg = OmegaConf.create({
            "num_workers": 0,
            "sample_rate": sr,
            "batch_size": 64,
            "device": device,
            "verbose": True,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(tmpdir),
                "oracle_vad": False,
                "oracle_num_speakers": False,
                "ignore_overlap": False,
                "collar": 0.25,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.875,
                        "onset": 0.4,
                        "offset": 0.35,
                        "pad_onset": 0.05,
                        "pad_offset": -0.1,
                        "min_duration_on": 0.2,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.1],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": True,
                        "max_num_speakers": num_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    },
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephony",
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [0.7],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_eval_settings": [[0.25, False]],
                    },
                },
            },
        })

        # Disable struct mode so NeMo can access top-level keys like 'device'
        # from any sub-config level without OmegaConf validation errors
        OmegaConf.set_struct(cfg, False)

        try:
            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()

            # Read RTTM output
            rttm_path = tmpdir / "pred_rttms" / "audio.rttm"
            segments = _parse_rttm(rttm_path)
            if segments:
                return segments
        except Exception as e:
            print(f"[NeMo] MSDD failed ({e}), falling back to clustering-only...")

        # Fallback: simpler clustering diarizer
        try:
            cfg.diarizer.msdd_model = OmegaConf.create({"model_path": None})
            OmegaConf.set_struct(cfg, False)
            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()
            rttm_path = tmpdir / "pred_rttms" / "audio.rttm"
            return _parse_rttm(rttm_path)
        except Exception as e2:
            import traceback
            traceback.print_exc()
            print(f"[NeMo] Clustering fallback also failed: {e2}")
            return []


def _parse_rttm(rttm_path: Path) -> list[dict]:
    """Parse NeMo RTTM output file into segment dicts."""
    segments = []
    if not rttm_path.exists():
        return segments

    for line in rttm_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        segments.append({
            "speaker": speaker,
            "start": start,
            "end": start + duration,
        })

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# NeMo Full Pipeline
# ---------------------------------------------------------------------------


def run_pipeline_nemo(
    channel_a_bytes: bytes,
    channel_b_bytes: bytes,
    mixed_bytes: bytes,
    session_name: str,
    num_speakers: int = 2,
    denoise_enabled: bool = True,
    denoise_prop: float = 0.6,
    fade_ms: int = 20,
    peak_db: float = -1.0,
) -> dict:
    """Pipeline D: NeMo MSDD/TitaNet + Channel-Aware Separation.

    Uses NeMo's state-of-the-art MSDD diarizer instead of pyannote,
    then applies the same channel-aware + Wiener masking separation.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    from src.pipeline import (
        load_and_preprocess,
        denoise_audio,
        postprocess_segments,
        extract_speaker_embeddings,
        assign_speakers,
        correct_segments_with_channels,
        find_overlaps,
        estimate_leakage,
        compute_spectral_profiles,
        channel_aware_separate,
        vad_gate_track,
        postprocess_track,
    )
    # Lazy import embedder for NeMo path (use SpeechBrain ECAPA separately)
    from speechbrain.inference.speaker import EncoderClassifier
    import torch

    print(f"\n{'='*60}")
    print(f"[Pipeline D: NeMo MSDD + Channel-Aware] {session_name}")
    print(f"{'='*60}")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Stage 0] Loading audio channels...")
    ch_a_audio, sr = load_and_preprocess(channel_a_bytes)
    ch_b_audio, _ = load_and_preprocess(channel_b_bytes)
    mixed_audio, _ = load_and_preprocess(mixed_bytes)
    print(f"  Duration: {len(mixed_audio)/sr:.1f}s @ {sr} Hz")

    if denoise_enabled:
        print("[Stage 1] Denoising signals...")
        mixed_clean = denoise_audio(mixed_audio, sr, denoise_prop)
        ch_a_clean = denoise_audio(ch_a_audio, sr, denoise_prop)
        ch_b_clean = denoise_audio(ch_b_audio, sr, denoise_prop)
    else:
        mixed_clean = mixed_audio
        ch_a_clean = ch_a_audio
        ch_b_clean = ch_b_audio

    print("[Stage 2-NeMo] Running NeMo MSDD diarization...")
    raw_segments = diarize_nemo(mixed_clean, sr, num_speakers)
    nemo_model = "nemo_msdd_titanet"
    speakers_found = set(s["speaker"] for s in raw_segments)
    print(f"  Found {len(speakers_found)} speakers, {len(raw_segments)} raw segments")

    if not raw_segments:
        print("[WARN] NeMo returned no segments; pipeline cannot continue.")
        return {
            "speaker_a": b"",
            "speaker_b": b"",
            "metadata": {"elapsed_s": round(time.time() - t0, 1), "error": "NeMo returned no segments"},
        }

    print("[Stage 3] Post-processing diarization...")
    segments = postprocess_segments(raw_segments, mixed_clean, sr)
    print(f"  Cleaned: {len(raw_segments)} -> {len(segments)} segments")

    print("[Stage 4] Loading SpeechBrain embedder + extracting embeddings...")
    embedder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    non_overlaps_raw, _ = find_overlaps(segments)
    speaker_embeddings = extract_speaker_embeddings(mixed_clean, sr, non_overlaps_raw, embedder)
    print(f"  Embeddings: {list(speaker_embeddings.keys())}")

    print("[Stage 5] Assigning speaker labels...")
    speaker_a_label, speaker_b_label = assign_speakers(
        segments, speaker_embeddings, ch_a_clean, ch_b_clean, sr,
    )
    print(f"  Speaker A = {speaker_a_label}, Speaker B = {speaker_b_label}")

    print("[Stage 6] Channel-informed diarization correction...")
    segments = correct_segments_with_channels(
        segments, ch_a_clean, ch_b_clean, sr, speaker_a_label, speaker_b_label,
    )

    non_overlaps, overlaps = find_overlaps(segments)
    total_dur = len(mixed_audio) / sr
    overlap_dur = sum(r["end"] - r["start"] for r in overlaps)
    overlap_pct = round((overlap_dur / total_dur * 100) if total_dur > 0 else 0, 1)
    print(f"  Non-overlap: {len(non_overlaps)} segs, Overlap: {len(overlaps)} regions ({overlap_pct}%)")

    print("[Stage 7] Estimating cross-channel leakage...")
    leak_a2b, leak_b2a = estimate_leakage(
        ch_a_clean, ch_b_clean, sr, non_overlaps, speaker_a_label, speaker_b_label,
    )

    print("[Stage 8] Building speaker spectral profiles...")
    profile_a, profile_b = compute_spectral_profiles(
        ch_a_clean, ch_b_clean, sr, non_overlaps, speaker_a_label, speaker_b_label,
    )

    print("[Stage 9] Channel-aware separation + Wiener overlap filtering...")
    tracks = channel_aware_separate(
        ch_a_clean, ch_b_clean, sr,
        non_overlaps, overlaps,
        speaker_a_label, speaker_b_label,
        leak_a2b, leak_b2a,
        profile_a, profile_b,
        fade_ms,
    )

    print("[Stage 10] VAD gating...")
    for key in ["speaker_a", "speaker_b"]:
        tracks[key] = vad_gate_track(tracks[key], sr)

    print("[Stage 11] Normalizing and encoding...")
    elapsed = round(time.time() - t0, 1)
    result = {
        "speaker_a": postprocess_track(tracks["speaker_a"], sr, peak_db),
        "speaker_b": postprocess_track(tracks["speaker_b"], sr, peak_db),
        "metadata": {
            "elapsed_s": elapsed,
            "segments": len(segments),
            "raw_segments": len(raw_segments),
            "overlap_pct": overlap_pct,
            "leak_a2b": leak_a2b,
            "leak_b2a": leak_b2a,
            "model": nemo_model,
        },
    }
    print(f"[Pipeline D DONE] {elapsed}s\n")
    return result
