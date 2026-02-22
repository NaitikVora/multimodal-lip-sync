"""Speaker separation pipeline — runs INSIDE Modal GPU container.

All imports in this file are container-side only (torch, pyannote, speechbrain, etc.).
Functions are called by the Modal Cls defined in modal_app.py.
"""

import io

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Stage 0: Audio Loading & Preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes, downmix to mono, resample to 16 kHz, normalize."""
    import librosa

    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    return data, sr


# ---------------------------------------------------------------------------
# Stage 1: Denoising
# ---------------------------------------------------------------------------


def denoise_audio(
    audio: np.ndarray, sr: int, prop_decrease: float = 0.6
) -> np.ndarray:
    """Reduce stationary noise with noisereduce. Skippable via config."""
    try:
        import noisereduce as nr

        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
    except Exception as e:
        print(f"[WARN] Denoising failed, using raw audio: {e}")
        return audio


# ---------------------------------------------------------------------------
# Stage 2: Speaker Diarization
# ---------------------------------------------------------------------------


def diarize(
    audio: np.ndarray, sr: int, diarizer, num_speakers: int = 2
) -> list[dict]:
    """Run pyannote diarization, return list of segment dicts.

    Each dict: {"speaker": str, "start": float, "end": float}
    Also tags overlap regions.
    """
    import torch

    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    input_dict = {"waveform": waveform, "sample_rate": sr}

    try:
        annotation = diarizer(input_dict, num_speakers=num_speakers)
    except Exception as e:
        print(f"[WARN] Diarization with num_speakers={num_speakers} failed: {e}")
        print("[WARN] Retrying without speaker count constraint...")
        annotation = diarizer(input_dict)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })

    segments.sort(key=lambda s: s["start"])
    return segments


def find_overlaps(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split segments into non-overlapping and overlapping regions.

    Returns (non_overlap_segments, overlap_regions).
    """
    if not segments:
        return [], []

    speakers = sorted(set(s["speaker"] for s in segments))
    if len(speakers) < 2:
        return segments, []

    # Build per-speaker timelines
    from collections import defaultdict

    speaker_intervals = defaultdict(list)
    for seg in segments:
        speaker_intervals[seg["speaker"]].append((seg["start"], seg["end"]))

    # Find overlapping time regions across any two speakers
    all_intervals = [(seg["start"], seg["end"]) for seg in segments]
    all_intervals.sort()

    overlaps = []
    non_overlaps = []

    # Use an event-based approach
    events = []
    for seg in segments:
        events.append((seg["start"], 1, seg["speaker"]))
        events.append((seg["end"], -1, seg["speaker"]))
    events.sort(key=lambda e: (e[0], e[1]))

    active = {}
    prev_time = 0.0

    for time, delta, speaker in events:
        if time > prev_time and active:
            active_speakers = [s for s, c in active.items() if c > 0]
            region = {"start": prev_time, "end": time, "speakers": active_speakers}
            if len(active_speakers) > 1:
                overlaps.append(region)
            elif len(active_speakers) == 1:
                non_overlaps.append({
                    "speaker": active_speakers[0],
                    "start": prev_time,
                    "end": time,
                })

        if delta == 1:
            active[speaker] = active.get(speaker, 0) + 1
        else:
            active[speaker] = active.get(speaker, 0) - 1
            if active[speaker] <= 0:
                del active[speaker]

        prev_time = time

    return non_overlaps, overlaps


# ---------------------------------------------------------------------------
# Stage 3: Speaker Embedding Extraction
# ---------------------------------------------------------------------------


def extract_speaker_embeddings(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    embedder,
    num_enrollment: int = 5,
    min_duration: float = 1.0,
) -> dict:
    """Extract centroid embedding for each speaker from their longest segments."""
    import torch

    speakers = sorted(set(s["speaker"] for s in segments))
    speaker_embeddings = {}

    for spk in speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        spk_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        spk_segs = [s for s in spk_segs if (s["end"] - s["start"]) >= min_duration]
        spk_segs = spk_segs[:num_enrollment]

        if not spk_segs:
            # Fall back to the longest segment regardless of duration
            spk_segs = sorted(
                [s for s in segments if s["speaker"] == spk],
                key=lambda s: s["end"] - s["start"],
                reverse=True,
            )[:1]

        embeddings = []
        for seg in spk_segs:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # < 100ms
                continue

            wav_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
            emb = embedder.encode_batch(wav_tensor)
            embeddings.append(emb.squeeze().cpu())

        if embeddings:
            centroid = torch.stack(embeddings).mean(dim=0)
            speaker_embeddings[spk] = centroid
        else:
            print(f"[WARN] No embeddings extracted for speaker {spk}")

    return speaker_embeddings


# ---------------------------------------------------------------------------
# Stage 4: Hybrid Audio Separation
# ---------------------------------------------------------------------------


def cosine_fade(length: int, fade_in: bool = True) -> np.ndarray:
    """Generate a cosine fade curve of given length."""
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, np.pi / 2, length)
    if fade_in:
        return np.sin(t).astype(np.float32)
    else:
        return np.cos(t).astype(np.float32)


def separate_overlap_chunk(
    chunk: np.ndarray,
    sr: int,
    separator,
    embedder,
    speaker_embeddings: dict,
    min_duration: float = 0.5,
) -> dict:
    """Run SepFormer on an overlap chunk and assign sources to speakers via embeddings."""
    import torch

    speakers = list(speaker_embeddings.keys())

    # Pad short chunks
    original_len = len(chunk)
    if original_len < int(min_duration * sr):
        pad_len = int(min_duration * sr) - original_len
        chunk = np.concatenate([chunk, np.zeros(pad_len, dtype=np.float32)])

    try:
        wav_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(
            next(separator.mods.parameters()).device
        )
        est_sources = separator.separate_batch(wav_tensor)
        # est_sources shape: (batch, time, num_sources)
        sources = est_sources.squeeze(0).cpu().numpy()

        if sources.ndim == 1:
            # Only one source returned — can't separate
            return {spk: chunk[:original_len] / len(speakers) for spk in speakers}

        num_sources = sources.shape[1] if sources.ndim == 2 else 1
    except Exception as e:
        print(f"[WARN] SepFormer failed on overlap chunk: {e}")
        return {spk: chunk[:original_len] / len(speakers) for spk in speakers}

    # Match sources to speakers via embedding cosine similarity
    assignment = {}
    used_sources = set()

    for spk in speakers:
        best_sim = -float("inf")
        best_idx = 0

        for src_idx in range(num_sources):
            if src_idx in used_sources:
                continue

            source = sources[:, src_idx] if sources.ndim == 2 else sources
            source_trimmed = source[:original_len]

            if np.max(np.abs(source_trimmed)) < 1e-6:
                continue

            wav_tensor_src = torch.from_numpy(source_trimmed).unsqueeze(0).float()
            try:
                emb = embedder.encode_batch(wav_tensor_src).squeeze().cpu()
                sim = torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0),
                    speaker_embeddings[spk].unsqueeze(0),
                ).item()
            except Exception:
                sim = 0.0

            if sim > best_sim:
                best_sim = sim
                best_idx = src_idx

        used_sources.add(best_idx)
        source = sources[:, best_idx] if sources.ndim == 2 else sources
        assignment[spk] = source[:original_len]

    return assignment


def hybrid_separate(
    audio: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    overlap_regions: list[dict],
    separator,
    embedder,
    speaker_embeddings: dict,
    fade_ms: int = 15,
) -> dict:
    """Build per-speaker tracks using gating for non-overlaps, SepFormer for overlaps."""
    speakers = sorted(speaker_embeddings.keys())
    total_samples = len(audio)
    fade_samples = int(fade_ms * sr / 1000)

    tracks = {spk: np.zeros(total_samples, dtype=np.float32) for spk in speakers}

    # Non-overlapping segments: gate original audio
    for seg in non_overlap_segments:
        spk = seg["speaker"]
        if spk not in tracks:
            continue

        start = int(seg["start"] * sr)
        end = min(int(seg["end"] * sr), total_samples)
        chunk = audio[start:end].copy()

        # Apply cosine fades
        fi_len = min(fade_samples, len(chunk) // 2)
        fo_len = min(fade_samples, len(chunk) // 2)
        if fi_len > 0:
            chunk[:fi_len] *= cosine_fade(fi_len, fade_in=True)
        if fo_len > 0:
            chunk[-fo_len:] *= cosine_fade(fo_len, fade_in=False)

        tracks[spk][start:end] += chunk

    # Overlapping segments: SepFormer + embedding matching
    for region in overlap_regions:
        start = int(region["start"] * sr)
        end = min(int(region["end"] * sr), total_samples)
        chunk = audio[start:end]

        if len(chunk) < int(0.05 * sr):  # < 50ms — too short
            continue

        assignment = separate_overlap_chunk(
            chunk, sr, separator, embedder, speaker_embeddings
        )

        for spk in speakers:
            if spk in assignment:
                source = assignment[spk]
                fi_len = min(fade_samples, len(source) // 2)
                fo_len = min(fade_samples, len(source) // 2)
                if fi_len > 0:
                    source[:fi_len] *= cosine_fade(fi_len, fade_in=True)
                if fo_len > 0:
                    source[-fo_len:] *= cosine_fade(fo_len, fade_in=False)
                seg_len = min(len(source), end - start)
                tracks[spk][start : start + seg_len] += source[:seg_len]

    return tracks


# ---------------------------------------------------------------------------
# Stage 5: Stitching & Post-processing
# ---------------------------------------------------------------------------


def postprocess_track(audio: np.ndarray, sr: int, peak_db: float = -1.0) -> bytes:
    """Normalize and encode a speaker track to 16-bit PCM WAV bytes."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10 ** (peak_db / 20)
        audio = audio * (target_peak / peak)

    audio = np.clip(audio, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Stage 6: Doctor vs Patient Assignment
# ---------------------------------------------------------------------------


def assign_roles(
    tracks: dict,
    segments: list[dict],
    speaker_embeddings: dict,
) -> tuple[str, str]:
    """Determine which diarization speaker label maps to doctor vs patient.

    Primary heuristic: first speaker = doctor.
    Fallback: speaker with less total speaking time = doctor.

    Returns (doctor_label, patient_label).
    """
    speakers = sorted(speaker_embeddings.keys())
    if len(speakers) < 2:
        return speakers[0] if speakers else "SPEAKER_00", "SPEAKER_01"

    # Find who speaks first
    first_speaker = None
    min_start = float("inf")
    for seg in segments:
        if seg["start"] < min_start:
            min_start = seg["start"]
            first_speaker = seg["speaker"]

    # Calculate total speaking time per speaker
    speaking_time = {spk: 0.0 for spk in speakers}
    for seg in segments:
        speaking_time[seg["speaker"]] = (
            speaking_time.get(seg["speaker"], 0.0) + seg["end"] - seg["start"]
        )

    doctor_label = first_speaker
    patient_label = [s for s in speakers if s != doctor_label][0]

    # Sanity check: if the "doctor" speaks 3x more than "patient", swap
    # (patients typically have longer descriptions)
    if doctor_label and patient_label:
        doc_time = speaking_time.get(doctor_label, 0)
        pat_time = speaking_time.get(patient_label, 0)
        if doc_time > 3 * pat_time and pat_time > 0:
            doctor_label, patient_label = patient_label, doctor_label

    return doctor_label, patient_label


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    audio_bytes: bytes,
    session_name: str,
    diarizer,
    embedder,
    separator,
    denoise_enabled: bool = True,
    denoise_prop: float = 0.6,
    num_speakers: int = 2,
    num_enrollment: int = 5,
    min_enrollment_duration: float = 1.0,
    fade_ms: int = 15,
    peak_db: float = -1.0,
) -> dict:
    """Execute the full speaker separation pipeline.

    Returns dict: {"original": bytes, "doctor": bytes, "patient": bytes}
    """
    print(f"\n{'='*60}")
    print(f"Processing: {session_name}")
    print(f"{'='*60}")

    # Stage 0
    print("[Stage 0] Loading and preprocessing audio...")
    audio, sr = load_and_preprocess(audio_bytes)
    print(f"  Audio: {len(audio)/sr:.1f}s @ {sr} Hz, {len(audio)} samples")

    # Keep original for output
    original_bytes = postprocess_track(audio, sr, peak_db)

    # Stage 1
    if denoise_enabled:
        print("[Stage 1] Denoising...")
        audio_clean = denoise_audio(audio, sr, denoise_prop)
    else:
        print("[Stage 1] Denoising skipped (disabled)")
        audio_clean = audio

    # Stage 2
    print("[Stage 2] Diarizing (finding who spoke when)...")
    segments = diarize(audio_clean, sr, diarizer, num_speakers)
    speakers_found = set(s["speaker"] for s in segments)
    print(f"  Found {len(speakers_found)} speakers, {len(segments)} segments")

    if len(speakers_found) < 2:
        print("[WARN] Only one speaker found — output tracks will be identical")

    # Identify overlaps
    non_overlaps, overlaps = find_overlaps(segments)
    total_dur = len(audio) / sr
    overlap_dur = sum(r["end"] - r["start"] for r in overlaps)
    print(f"  Non-overlap segments: {len(non_overlaps)}")
    print(f"  Overlap regions: {len(overlaps)} ({overlap_dur:.1f}s / {overlap_dur/total_dur*100:.1f}%)")

    # Stage 3
    print("[Stage 3] Extracting speaker embeddings...")
    speaker_embeddings = extract_speaker_embeddings(
        audio_clean, sr, non_overlaps, embedder, num_enrollment, min_enrollment_duration
    )
    print(f"  Embeddings extracted for: {list(speaker_embeddings.keys())}")

    # Stage 4
    print("[Stage 4] Hybrid separation (gate + SepFormer)...")
    tracks = hybrid_separate(
        audio_clean, sr, non_overlaps, overlaps,
        separator, embedder, speaker_embeddings, fade_ms,
    )

    # Stage 5
    print("[Stage 5] Post-processing tracks...")
    track_bytes = {}
    for spk, track in tracks.items():
        track_bytes[spk] = postprocess_track(track, sr, peak_db)

    # Stage 6
    print("[Stage 6] Assigning doctor vs patient roles...")
    doctor_label, patient_label = assign_roles(tracks, segments, speaker_embeddings)
    print(f"  Doctor = {doctor_label}, Patient = {patient_label}")

    result = {
        "original": original_bytes,
        "doctor": track_bytes.get(doctor_label, original_bytes),
        "patient": track_bytes.get(patient_label, original_bytes),
    }

    for key, wav in result.items():
        dur = len(wav) / (sr * 2)  # approximate (16-bit mono)
        print(f"  Output '{key}': {len(wav)/1024:.0f} KB")

    print(f"[DONE] {session_name} complete.\n")
    return result
