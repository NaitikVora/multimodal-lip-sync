"""DiariZen-based speaker diarization pipeline — runs INSIDE Modal GPU container.

DiariZen is a state-of-the-art end-to-end diarization system from the BUT/LDC
team that won VoxSRC 2023. It uses a Conformer-based encoder with a speaker
change point detection approach combined with speaker embeddings.

Key advantages over pyannote and NeMo:
  - End-to-end training for diarization (no separate VAD step)
  - Excellent handling of overlapping speech
  - VoxSRC 2023 winner — best-in-class DER on challenging data
  - ResNet293 CAM++ speaker embeddings
"""

import io
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# DiariZen Diarization
# ---------------------------------------------------------------------------


def diarize_diarizen(
    audio: np.ndarray,
    sr: int,
    num_speakers: int = 2,
) -> list[dict]:
    """Run DiariZen → wespeaker fallback.

    Returns [{"speaker": str, "start": float, "end": float}, ...]
    """
    try:
        return _diarize_diarizen_native(audio, sr, num_speakers)
    except ImportError as e:
        print(f"[DiariZen] Package not available ({e}), trying wespeaker fallback...")

    try:
        return _diarize_wespeaker_fallback(audio, sr, num_speakers)
    except ImportError as e:
        print(f"[DiariZen] wespeaker not available ({e})")
        return []
    except Exception as e:
        print(f"[DiariZen] fallback failed: {e}")
        return []


def _diarize_diarizen_native(
    audio: np.ndarray,
    sr: int,
    num_speakers: int,
) -> list[dict]:
    """DiariZen native interface via the diarizen package."""
    import torch
    from diarizen.pipelines import DiarizationPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, audio, sr)

    try:
        pipeline = DiarizationPipeline.from_pretrained(
            "BUT-FIT/DiariZen",
            use_auth_token=None,
            device=device,
        )
        diarization = pipeline(
            tmp_path,
            num_speakers=num_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        segments.sort(key=lambda s: s["start"])
        return segments
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _diarize_wespeaker_fallback(
    audio: np.ndarray,
    sr: int,
    num_speakers: int,
) -> list[dict]:
    """Fallback: wespeaker (CAM++) + sklearn AgglomerativeClustering.

    Uses state-of-the-art CAM++ speaker embeddings with sliding-window
    segmentation and spectral clustering for a strong fallback that
    approximates DiariZen's quality.
    """
    import torch
    import wespeaker

    # Load CAM++ model from wespeaker — keep on CPU to avoid device
    # mismatch since wespeaker's extract_embedding creates CPU tensors
    model = wespeaker.load_model("english")
    if hasattr(model, 'set_gpu'):
        model.set_gpu(-1)  # force CPU

    # Sliding window segmentation
    win_s = 1.5
    shift_s = 0.5
    win_samples = int(win_s * sr)
    shift_samples = int(shift_s * sr)

    embeddings = []
    timestamps = []

    for start in range(0, len(audio) - win_samples, shift_samples):
        end = start + win_samples
        chunk = audio[start:end]

        # Use file-based extraction to avoid tensor device mismatch
        chunk_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk_path = tmp.name
                sf.write(chunk_path, chunk, sr)
            if hasattr(model, 'extract_embedding'):
                emb = model.extract_embedding(chunk_path)
            else:
                wav_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.extract_embedding_from_audio(wav_tensor, sr)
        finally:
            if chunk_path:
                Path(chunk_path).unlink(missing_ok=True)

        if isinstance(emb, torch.Tensor):
            embeddings.append(emb.squeeze().cpu().numpy())
        else:
            embeddings.append(np.asarray(emb).squeeze())
        timestamps.append((start / sr, end / sr))

    if not embeddings:
        return []

    # Spectral clustering
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    X = np.stack(embeddings)
    # L2-normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    # Convert to segments
    segments = []
    for i, (start, end) in enumerate(timestamps):
        segments.append({
            "speaker": f"SPEAKER_{labels[i]:02d}",
            "start": start,
            "end": end,
        })

    # Merge consecutive same-speaker segments
    segments.sort(key=lambda s: s["start"])
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] < 0.5:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return merged


# ---------------------------------------------------------------------------
# DiariZen Full Pipeline
# ---------------------------------------------------------------------------


def run_pipeline_diarizen(
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
    """Pipeline E: DiariZen (VoxSRC winner) + Channel-Aware Separation.

    Uses DiariZen's end-to-end diarization for best-in-class speaker
    detection, then applies channel-aware separation with Wiener masking.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    from speechbrain.inference.speaker import EncoderClassifier
    import torch
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

    print(f"\n{'='*60}")
    print(f"[Pipeline E: DiariZen + Channel-Aware] {session_name}")
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

    print("[Stage 2-DiariZen] Running DiariZen diarization...")
    raw_segments = diarize_diarizen(mixed_clean, sr, num_speakers)
    diarizen_model = "diarizen_wespeaker_campp"
    speakers_found = set(s["speaker"] for s in raw_segments)
    print(f"  Found {len(speakers_found)} speakers, {len(raw_segments)} raw segments")

    if not raw_segments:
        print("[WARN] DiariZen returned no segments; pipeline cannot continue.")
        return {
            "speaker_a": b"",
            "speaker_b": b"",
            "metadata": {"elapsed_s": round(time.time() - t0, 1), "error": "DiariZen returned no segments"},
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
            "model": diarizen_model,
        },
    }
    print(f"[Pipeline E DONE] {elapsed}s\n")
    return result
