"""Speaker separation pipeline — runs INSIDE Modal GPU container.

All imports in this file are container-side only (torch, pyannote, speechbrain, etc.).
Functions are called by the Modal Cls defined in modal_app.py.

Pipeline stages:
  0. Load & preprocess (mono, 16 kHz, normalize)
  1. Denoise (noisereduce with tuned parameters)
  2. Diarize (pyannote 3.1, constrained to 2 speakers)
  3. Post-process diarization segments (merge, clean, smooth)
  4. Extract duration-weighted speaker embeddings (ECAPA-TDNN)
  5. Assign speaker labels (first speaker + channel energy validation)
  6. Separate audio (channel-aware or SepFormer for overlaps)
  7. Post-process output tracks (normalize, encode)
"""

import io
import time

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
# Stage 1: Denoising (improved parameters)
# ---------------------------------------------------------------------------


def denoise_audio(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.6,
    n_fft: int = 2048,
    freq_mask_smooth_hz: int = 500,
    time_mask_smooth_ms: int = 50,
) -> np.ndarray:
    """Reduce stationary noise with noisereduce using tuned parameters.

    Improvements over defaults:
    - n_fft=2048 gives better frequency resolution for speech
    - freq_mask_smooth_hz=500 prevents harsh spectral artifacts
    - time_mask_smooth_ms=50 prevents choppy temporal gating
    """
    try:
        import noisereduce as nr

        return nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
        )
    except Exception as e:
        print(f"[WARN] Denoising failed, using raw audio: {e}")
        return audio


# ---------------------------------------------------------------------------
# Stage 2: Speaker Diarization
# ---------------------------------------------------------------------------


def diarize(
    audio: np.ndarray, sr: int, diarizer, num_speakers: int = 2
) -> list[dict]:
    """Run pyannote diarization, return list of segment dicts."""
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


# ---------------------------------------------------------------------------
# Stage 3: Diarization Post-Processing (NEW — major accuracy improvement)
# ---------------------------------------------------------------------------


def postprocess_segments(
    segments: list[dict],
    min_segment_duration: float = 0.25,
    max_gap_merge: float = 0.3,
    min_island_duration: float = 0.15,
) -> list[dict]:
    """Clean up raw diarization output for more accurate speaker separation.

    Four-pass algorithm:
      1. Merge same-speaker segments separated by short gaps (< max_gap_merge)
      2. Remove short "island" segments surrounded by another speaker
      3. Absorb remaining short segments (< min_segment_duration) into neighbors
      4. Final same-speaker merge pass to consolidate

    This dramatically reduces fragmented segments that cause choppy output.
    """
    if len(segments) < 2:
        return segments

    segments = sorted(segments, key=lambda s: s["start"])

    # Pass 1: Merge same-speaker segments with small gaps
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= max_gap_merge:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    # Pass 2: Remove short isolated "island" segments
    # An island is a short segment of Speaker X surrounded by Speaker Y on both sides
    if len(merged) >= 3:
        cleaned = [merged[0]]
        for i in range(1, len(merged) - 1):
            seg = merged[i]
            duration = seg["end"] - seg["start"]
            prev_spk = cleaned[-1]["speaker"]
            next_spk = merged[i + 1]["speaker"]

            if (
                duration < min_island_duration
                and prev_spk == next_spk
                and prev_spk != seg["speaker"]
            ):
                # Island detected — absorb into surrounding speaker
                cleaned[-1]["end"] = max(cleaned[-1]["end"], seg["end"])
            else:
                cleaned.append(seg.copy())
        cleaned.append(merged[-1].copy())
    else:
        cleaned = merged

    # Pass 3: Absorb segments shorter than min_segment_duration into neighbors
    result = []
    for seg in cleaned:
        duration = seg["end"] - seg["start"]
        if duration >= min_segment_duration:
            result.append(seg)
        elif result:
            # Absorb into previous segment (extend its end boundary)
            result[-1]["end"] = max(result[-1]["end"], seg["end"])
        # If no previous segment exists, drop silently (leading micro-segment)

    if not result:
        return cleaned  # Safety fallback

    # Pass 4: Final same-speaker merge (catches segments made adjacent by passes 2-3)
    final = [result[0].copy()]
    for seg in result[1:]:
        prev = final[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            final.append(seg.copy())

    return final


# ---------------------------------------------------------------------------
# Overlap Detection
# ---------------------------------------------------------------------------


def find_overlaps(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split segments into non-overlapping and overlapping regions.

    Uses an event-based sweep-line algorithm to precisely identify regions
    where multiple speakers are active simultaneously.
    """
    if not segments:
        return [], []

    speakers = sorted(set(s["speaker"] for s in segments))
    if len(speakers) < 2:
        return segments, []

    overlaps = []
    non_overlaps = []

    events = []
    for seg in segments:
        events.append((seg["start"], 1, seg["speaker"]))
        events.append((seg["end"], -1, seg["speaker"]))
    events.sort(key=lambda e: (e[0], e[1]))

    active: dict[str, int] = {}
    prev_time = 0.0

    for time_val, delta, speaker in events:
        if time_val > prev_time and active:
            active_speakers = [s for s, c in active.items() if c > 0]
            region = {"start": prev_time, "end": time_val, "speakers": active_speakers}
            if len(active_speakers) > 1:
                overlaps.append(region)
            elif len(active_speakers) == 1:
                non_overlaps.append({
                    "speaker": active_speakers[0],
                    "start": prev_time,
                    "end": time_val,
                })

        if delta == 1:
            active[speaker] = active.get(speaker, 0) + 1
        else:
            active[speaker] = active.get(speaker, 0) - 1
            if active[speaker] <= 0:
                del active[speaker]

        prev_time = time_val

    return non_overlaps, overlaps


# ---------------------------------------------------------------------------
# Stage 4: Speaker Embedding Extraction (improved)
# ---------------------------------------------------------------------------


def extract_speaker_embeddings(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    embedder,
    num_enrollment: int = 8,
    min_duration: float = 0.5,
) -> dict:
    """Extract duration-weighted centroid embedding for each speaker.

    Improvements over simple averaging:
    - Uses more enrollment segments (8 vs 5) for better representation
    - Lower minimum duration threshold (0.5s vs 1.0s) to include more data
    - Duration-weighted centroid: longer segments contribute more to the centroid,
      producing more stable speaker representations
    """
    import torch

    speakers = sorted(set(s["speaker"] for s in segments))
    speaker_embeddings = {}

    for spk in speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        spk_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        eligible = [s for s in spk_segs if (s["end"] - s["start"]) >= min_duration]

        if not eligible:
            # Fall back to the longest segments available
            eligible = spk_segs[:3]
        else:
            eligible = eligible[:num_enrollment]

        embeddings = []
        weights = []
        for seg in eligible:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:
                continue

            wav_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
            emb = embedder.encode_batch(wav_tensor)
            embeddings.append(emb.squeeze().cpu())
            weights.append(seg["end"] - seg["start"])

        if embeddings:
            # Duration-weighted centroid (longer segments = more reliable)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()
            stacked = torch.stack(embeddings)
            centroid = (stacked * weights_tensor.unsqueeze(1)).sum(dim=0)
            speaker_embeddings[spk] = centroid
        else:
            print(f"[WARN] No embeddings extracted for speaker {spk}")

    return speaker_embeddings


# ---------------------------------------------------------------------------
# Stage 5: Speaker Assignment (improved)
# ---------------------------------------------------------------------------


def assign_speakers(
    segments: list[dict],
    speaker_embeddings: dict,
    channel_a_audio: np.ndarray | None = None,
    channel_b_audio: np.ndarray | None = None,
    sr: int = 16000,
) -> tuple[str, str]:
    """Determine which diarization label maps to Speaker A vs Speaker B.

    Primary: first speaker detected = Speaker A.
    Validation: if channel audio is available, use channel-energy correlation
    to verify and potentially correct the assignment.
    """
    speakers = sorted(speaker_embeddings.keys())
    if len(speakers) < 2:
        return speakers[0] if speakers else "SPEAKER_00", "SPEAKER_01"

    # Primary: first speaker = Speaker A
    first_speaker = None
    min_start = float("inf")
    for seg in segments:
        if seg["start"] < min_start:
            min_start = seg["start"]
            first_speaker = seg["speaker"]

    speaker_a_label = first_speaker
    speaker_b_label = [s for s in speakers if s != speaker_a_label][0]

    # Channel-energy validation: check if channel affinity is consistent
    if channel_a_audio is not None and channel_b_audio is not None:
        speaker_a_label, speaker_b_label = _validate_with_channel_energy(
            segments, channel_a_audio, channel_b_audio, sr,
            speaker_a_label, speaker_b_label,
        )

    return speaker_a_label, speaker_b_label


def _validate_with_channel_energy(
    segments: list[dict],
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    sr: int,
    speaker_a_label: str,
    speaker_b_label: str,
) -> tuple[str, str]:
    """Use per-segment channel energy to validate speaker-channel mapping.

    If speaker_a consistently has more energy on channel_b (and vice versa),
    the labels should be swapped. This corrects the common case where the
    first-speaker heuristic assigns labels opposite to the actual mic placement.
    """
    ch_a_ratio = {speaker_a_label: [], speaker_b_label: []}

    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        if end <= start or end > len(channel_a) or end > len(channel_b):
            continue

        e_a = np.sqrt(np.mean(channel_a[start:end] ** 2) + 1e-10)
        e_b = np.sqrt(np.mean(channel_b[start:end] ** 2) + 1e-10)
        ratio = e_a / (e_a + e_b + 1e-10)

        spk = seg["speaker"]
        if spk in ch_a_ratio:
            ch_a_ratio[spk].append(ratio)

    # Compute mean channel-A affinity for each speaker
    a_mean = np.mean(ch_a_ratio[speaker_a_label]) if ch_a_ratio[speaker_a_label] else 0.5
    b_mean = np.mean(ch_a_ratio[speaker_b_label]) if ch_a_ratio[speaker_b_label] else 0.5

    # If Speaker A is consistently louder on channel B (ratio < 0.4) while
    # Speaker B is louder on channel A (ratio > 0.6), swap the assignment
    if a_mean < 0.4 and b_mean > 0.6:
        print(f"  [Channel validation] Swapping labels: "
              f"A_ch_a={a_mean:.2f}, B_ch_a={b_mean:.2f}")
        return speaker_b_label, speaker_a_label

    return speaker_a_label, speaker_b_label


# ---------------------------------------------------------------------------
# Stage 6a: Channel-Aware Audio Separation (Pipeline B)
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


def _apply_faded_chunk(
    track: np.ndarray, start: int, end: int, chunk: np.ndarray, fade_samples: int
):
    """Apply a chunk with cosine fades into the track buffer using overlap-add."""
    if len(chunk) == 0:
        return
    fi_len = min(fade_samples, len(chunk) // 2)
    fo_len = min(fade_samples, len(chunk) // 2)
    if fi_len > 0:
        chunk[:fi_len] *= cosine_fade(fi_len, fade_in=True)
    if fo_len > 0:
        chunk[-fo_len:] *= cosine_fade(fo_len, fade_in=False)
    actual_end = min(end, start + len(chunk), len(track))
    actual_len = actual_end - start
    if actual_len > 0:
        track[start:actual_end] += chunk[:actual_len]


def channel_aware_separate(
    channel_a_audio: np.ndarray,
    channel_b_audio: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    overlap_regions: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    fade_ms: int = 20,
) -> dict:
    """Build per-speaker tracks using the appropriate mic channel for each segment."""
    total_samples = len(channel_a_audio)
    fade_samples = int(fade_ms * sr / 1000)

    track_a = np.zeros(total_samples, dtype=np.float32)
    track_b = np.zeros(total_samples, dtype=np.float32)

    label_to_role = {speaker_a_label: "speaker_a", speaker_b_label: "speaker_b"}

    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = min(int(seg["end"] * sr), total_samples)
        if end <= start:
            continue
        role = label_to_role.get(seg["speaker"])

        if role == "speaker_a":
            _apply_faded_chunk(track_a, start, end, channel_a_audio[start:end].copy(), fade_samples)
        elif role == "speaker_b":
            _apply_faded_chunk(track_b, start, end, channel_b_audio[start:end].copy(), fade_samples)

    for region in overlap_regions:
        start = int(region["start"] * sr)
        end = min(int(region["end"] * sr), total_samples)
        if end - start < int(0.05 * sr):
            continue
        _apply_faded_chunk(track_a, start, end, channel_a_audio[start:end].copy(), fade_samples)
        _apply_faded_chunk(track_b, start, end, channel_b_audio[start:end].copy(), fade_samples)

    return {"speaker_a": track_a, "speaker_b": track_b}


# ---------------------------------------------------------------------------
# Stage 6b: SepFormer-based Separation for Overlaps (Pipeline C)
# ---------------------------------------------------------------------------


def _separate_overlap_sepformer(
    mixed_chunk: np.ndarray,
    sr: int,
    separator,
    embedder,
    speaker_embeddings: dict,
    speaker_a_label: str,
    speaker_b_label: str,
    context_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SepFormer on an overlap chunk and assign sources via embedding matching.

    If context_audio is provided, it is prepended to the chunk for better
    separation quality, then stripped from the output.
    """
    import torch
    import torch.nn.functional as F

    min_samples = int(0.5 * sr)
    original_len = len(mixed_chunk)

    # Prepend context for better separation (SepFormer benefits from context)
    context_len = 0
    if context_audio is not None and len(context_audio) > 0:
        mixed_chunk = np.concatenate([context_audio, mixed_chunk])
        context_len = len(context_audio)

    if len(mixed_chunk) < min_samples:
        mixed_chunk = np.pad(mixed_chunk, (0, min_samples - len(mixed_chunk)))

    wav_tensor = torch.from_numpy(mixed_chunk).unsqueeze(0).float()
    sep_sources = separator.separate_batch(wav_tensor)

    if sep_sources.dim() == 3:
        src1 = sep_sources[0, :, 0].cpu().numpy()
        src2 = sep_sources[0, :, 1].cpu().numpy()
    else:
        src1 = sep_sources[0].cpu().numpy()
        src2 = sep_sources[1].cpu().numpy() if sep_sources.shape[0] > 1 else src1.copy()

    # Strip context and padding
    src1 = src1[context_len:context_len + original_len]
    src2 = src2[context_len:context_len + original_len]

    # Assign sources to speakers via embedding cosine similarity
    a_emb = speaker_embeddings.get(speaker_a_label)
    b_emb = speaker_embeddings.get(speaker_b_label)

    if a_emb is None or b_emb is None:
        return src1, src2

    # Only compute embeddings if chunks are long enough
    if len(src1) < sr * 0.1 or len(src2) < sr * 0.1:
        return src1, src2

    emb1 = embedder.encode_batch(torch.from_numpy(src1).unsqueeze(0).float()).squeeze().cpu()
    emb2 = embedder.encode_batch(torch.from_numpy(src2).unsqueeze(0).float()).squeeze().cpu()

    # Use both speaker embeddings for more robust permutation resolution
    sim_1a = F.cosine_similarity(emb1.unsqueeze(0), a_emb.unsqueeze(0)).item()
    sim_2a = F.cosine_similarity(emb2.unsqueeze(0), a_emb.unsqueeze(0)).item()
    sim_1b = F.cosine_similarity(emb1.unsqueeze(0), b_emb.unsqueeze(0)).item()
    sim_2b = F.cosine_similarity(emb2.unsqueeze(0), b_emb.unsqueeze(0)).item()

    # Hungarian-style: pick assignment that maximizes total similarity
    score_normal = sim_1a + sim_2b  # src1=A, src2=B
    score_swapped = sim_2a + sim_1b  # src2=A, src1=B

    if score_normal >= score_swapped:
        return src1, src2
    else:
        return src2, src1


def sepformer_separate(
    channel_a_audio: np.ndarray,
    channel_b_audio: np.ndarray,
    mixed_audio: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    overlap_regions: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    separator,
    embedder,
    speaker_embeddings: dict,
    fade_ms: int = 20,
    context_s: float = 0.25,
) -> dict:
    """Hybrid separation: channel-aware for non-overlaps, SepFormer for overlaps.

    Improvements:
    - Context padding around overlap chunks for better SepFormer quality
    - Hungarian-style permutation resolution using both speaker embeddings
    """
    total_samples = len(channel_a_audio)
    fade_samples = int(fade_ms * sr / 1000)
    context_samples = int(context_s * sr)

    track_a = np.zeros(total_samples, dtype=np.float32)
    track_b = np.zeros(total_samples, dtype=np.float32)

    label_to_role = {speaker_a_label: "speaker_a", speaker_b_label: "speaker_b"}

    # Non-overlaps: extract from appropriate mic channel
    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = min(int(seg["end"] * sr), total_samples)
        if end <= start:
            continue
        role = label_to_role.get(seg["speaker"])

        if role == "speaker_a":
            _apply_faded_chunk(track_a, start, end, channel_a_audio[start:end].copy(), fade_samples)
        elif role == "speaker_b":
            _apply_faded_chunk(track_b, start, end, channel_b_audio[start:end].copy(), fade_samples)

    # Overlaps: run SepFormer with context + embedding matching
    for region in overlap_regions:
        start = int(region["start"] * sr)
        end = min(int(region["end"] * sr), total_samples)
        if end - start < int(0.05 * sr):
            continue

        mixed_chunk = mixed_audio[start:end]

        # Extract context from before the overlap for better separation
        ctx_start = max(0, start - context_samples)
        context = mixed_audio[ctx_start:start] if ctx_start < start else None

        try:
            src_a, src_b = _separate_overlap_sepformer(
                mixed_chunk, sr, separator, embedder,
                speaker_embeddings, speaker_a_label, speaker_b_label,
                context_audio=context,
            )
            _apply_faded_chunk(track_a, start, start + len(src_a), src_a.copy(), fade_samples)
            _apply_faded_chunk(track_b, start, start + len(src_b), src_b.copy(), fade_samples)
        except Exception as e:
            print(f"[WARN] SepFormer failed on overlap at {region['start']:.1f}s: {e}")
            # Fallback: use channel audio
            _apply_faded_chunk(track_a, start, end, channel_a_audio[start:end].copy(), fade_samples)
            _apply_faded_chunk(track_b, start, end, channel_b_audio[start:end].copy(), fade_samples)

    return {"speaker_a": track_a, "speaker_b": track_b}


# ---------------------------------------------------------------------------
# Stage 7: Post-processing
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
# Shared Stages: common preprocessing for both Pipeline B and C
# ---------------------------------------------------------------------------


def _run_shared_stages(
    channel_a_bytes: bytes,
    channel_b_bytes: bytes,
    mixed_bytes: bytes,
    session_name: str,
    diarizer,
    embedder,
    denoise_enabled: bool = True,
    denoise_prop: float = 0.6,
    num_speakers: int = 2,
    num_enrollment: int = 8,
    min_enrollment_duration: float = 0.5,
    min_segment_duration: float = 0.25,
    max_gap_merge: float = 0.3,
    min_island_duration: float = 0.15,
) -> dict:
    """Run Stages 0-5 (preprocessing through speaker assignment), return state."""
    print("[Stage 0] Loading audio channels...")
    ch_a_audio, sr = load_and_preprocess(channel_a_bytes)
    ch_b_audio, _ = load_and_preprocess(channel_b_bytes)
    mixed_audio, _ = load_and_preprocess(mixed_bytes)
    print(f"  Duration: {len(mixed_audio)/sr:.1f}s @ {sr} Hz")

    if denoise_enabled:
        print("[Stage 1] Denoising mixed signal for diarization...")
        mixed_clean = denoise_audio(mixed_audio, sr, denoise_prop)
    else:
        mixed_clean = mixed_audio

    print("[Stage 2] Diarizing (finding who spoke when)...")
    raw_segments = diarize(mixed_clean, sr, diarizer, num_speakers)
    speakers_found = set(s["speaker"] for s in raw_segments)
    print(f"  Found {len(speakers_found)} speakers, {len(raw_segments)} raw segments")

    print("[Stage 3] Post-processing diarization segments...")
    segments = postprocess_segments(
        raw_segments, min_segment_duration, max_gap_merge, min_island_duration,
    )
    print(f"  Cleaned: {len(raw_segments)} → {len(segments)} segments "
          f"(removed {len(raw_segments) - len(segments)} fragments)")

    non_overlaps, overlaps = find_overlaps(segments)
    total_dur = len(mixed_audio) / sr
    overlap_dur = sum(r["end"] - r["start"] for r in overlaps)
    overlap_pct = (overlap_dur / total_dur * 100) if total_dur > 0 else 0
    print(f"  Non-overlap: {len(non_overlaps)} segments")
    print(f"  Overlap: {len(overlaps)} regions ({overlap_dur:.1f}s / {overlap_pct:.1f}%)")

    print("[Stage 4] Extracting speaker embeddings (duration-weighted)...")
    speaker_embeddings = extract_speaker_embeddings(
        mixed_clean, sr, non_overlaps, embedder, num_enrollment, min_enrollment_duration
    )
    print(f"  Embeddings: {list(speaker_embeddings.keys())}")

    print("[Stage 5] Assigning speaker labels...")
    speaker_a_label, speaker_b_label = assign_speakers(
        segments, speaker_embeddings, ch_a_audio, ch_b_audio, sr,
    )
    print(f"  Speaker A = {speaker_a_label}, Speaker B = {speaker_b_label}")

    return {
        "ch_a_audio": ch_a_audio,
        "ch_b_audio": ch_b_audio,
        "mixed_audio": mixed_audio,
        "mixed_clean": mixed_clean,
        "sr": sr,
        "segments": segments,
        "raw_segments": raw_segments,
        "non_overlaps": non_overlaps,
        "overlaps": overlaps,
        "speaker_embeddings": speaker_embeddings,
        "speaker_a_label": speaker_a_label,
        "speaker_b_label": speaker_b_label,
        "overlap_pct": round(overlap_pct, 1),
        "num_segments": len(segments),
        "num_raw_segments": len(raw_segments),
    }


# ---------------------------------------------------------------------------
# Pipeline B: ML Diarization + Channel-Aware Extraction
# ---------------------------------------------------------------------------


def run_pipeline(
    channel_a_bytes: bytes,
    channel_b_bytes: bytes,
    mixed_bytes: bytes,
    session_name: str,
    diarizer,
    embedder,
    separator,
    denoise_enabled: bool = True,
    denoise_prop: float = 0.6,
    num_speakers: int = 2,
    num_enrollment: int = 8,
    min_enrollment_duration: float = 0.5,
    fade_ms: int = 20,
    peak_db: float = -1.0,
) -> dict:
    """Pipeline B: channel-aware speaker separation.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline B: Channel-Aware] {session_name}")
    print(f"{'='*60}")

    t0 = time.time()
    state = _run_shared_stages(
        channel_a_bytes, channel_b_bytes, mixed_bytes, session_name,
        diarizer, embedder, denoise_enabled, denoise_prop,
        num_speakers, num_enrollment, min_enrollment_duration,
    )

    print("[Stage 6] Channel-aware separation...")
    tracks = channel_aware_separate(
        state["ch_a_audio"], state["ch_b_audio"], state["sr"],
        state["non_overlaps"], state["overlaps"],
        state["speaker_a_label"], state["speaker_b_label"],
        fade_ms,
    )

    print("[Stage 7] Post-processing...")
    elapsed = round(time.time() - t0, 1)
    result = {
        "speaker_a": postprocess_track(tracks["speaker_a"], state["sr"], peak_db),
        "speaker_b": postprocess_track(tracks["speaker_b"], state["sr"], peak_db),
        "metadata": {
            "elapsed_s": elapsed,
            "segments": state["num_segments"],
            "raw_segments": state["num_raw_segments"],
            "overlap_pct": state["overlap_pct"],
        },
    }

    for key in ["speaker_a", "speaker_b"]:
        print(f"  Output '{key}': {len(result[key])/1024:.0f} KB")
    print(f"[Pipeline B DONE] {elapsed}s\n")
    return result


# ---------------------------------------------------------------------------
# Pipeline C: ML Diarization + SepFormer on Overlaps
# ---------------------------------------------------------------------------


def run_pipeline_sepformer(
    channel_a_bytes: bytes,
    channel_b_bytes: bytes,
    mixed_bytes: bytes,
    session_name: str,
    diarizer,
    embedder,
    separator,
    denoise_enabled: bool = True,
    denoise_prop: float = 0.6,
    num_speakers: int = 2,
    num_enrollment: int = 8,
    min_enrollment_duration: float = 0.5,
    fade_ms: int = 20,
    peak_db: float = -1.0,
    context_s: float = 0.25,
) -> dict:
    """Pipeline C: SepFormer-based separation for overlap regions.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline C: SepFormer] {session_name}")
    print(f"{'='*60}")

    t0 = time.time()
    state = _run_shared_stages(
        channel_a_bytes, channel_b_bytes, mixed_bytes, session_name,
        diarizer, embedder, denoise_enabled, denoise_prop,
        num_speakers, num_enrollment, min_enrollment_duration,
    )

    print("[Stage 6] SepFormer separation on overlaps...")
    tracks = sepformer_separate(
        state["ch_a_audio"], state["ch_b_audio"], state["mixed_audio"], state["sr"],
        state["non_overlaps"], state["overlaps"],
        state["speaker_a_label"], state["speaker_b_label"],
        separator, embedder, state["speaker_embeddings"],
        fade_ms, context_s,
    )

    print("[Stage 7] Post-processing...")
    elapsed = round(time.time() - t0, 1)
    result = {
        "speaker_a": postprocess_track(tracks["speaker_a"], state["sr"], peak_db),
        "speaker_b": postprocess_track(tracks["speaker_b"], state["sr"], peak_db),
        "metadata": {
            "elapsed_s": elapsed,
            "segments": state["num_segments"],
            "raw_segments": state["num_raw_segments"],
            "overlap_pct": state["overlap_pct"],
        },
    }

    for key in ["speaker_a", "speaker_b"]:
        print(f"  Output '{key}': {len(result[key])/1024:.0f} KB")
    print(f"[Pipeline C DONE] {elapsed}s\n")
    return result
