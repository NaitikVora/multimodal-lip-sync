"""Speaker separation pipeline — runs INSIDE Modal GPU container.

All imports in this file are container-side only (torch, pyannote, speechbrain, etc.).
Functions are called by the Modal Cls defined in modal_app.py.

Pipeline stages:
  0. Load & preprocess (mono, 16 kHz, normalize)
  1. Denoise (noisereduce with tuned parameters)
  2. Diarize (pyannote 3.1, constrained to 2 speakers)
  3. Post-process diarization (merge, clean, smooth, boundary refinement)
  4. Channel-informed diarization correction
  5. Extract duration-weighted speaker embeddings (ECAPA-TDNN)
  6. Assign speaker labels (first speaker + channel energy validation)
  7. Estimate cross-channel leakage coefficients
  8. Build speaker spectral profiles for Wiener masking
  9. Separate audio:
     - Non-overlaps: primary channel + spectral leakage suppression
     - Overlaps: Wiener filtering with speaker profiles (Pipeline B)
                 or SepFormer + Wiener refinement (Pipeline C)
 10. VAD gating: silence non-speech residuals in each track
 11. Normalize and encode
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
# Stage 1: Denoising
# ---------------------------------------------------------------------------


def denoise_audio(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.6,
    n_fft: int = 2048,
    freq_mask_smooth_hz: int = 500,
    time_mask_smooth_ms: int = 50,
) -> np.ndarray:
    """Reduce stationary noise with tuned noisereduce parameters."""
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
# Stage 3: Diarization Post-Processing
# ---------------------------------------------------------------------------


def refine_boundaries(
    segments: list[dict],
    audio: np.ndarray,
    sr: int,
    snap_ms: int = 30,
) -> list[dict]:
    """Snap segment boundaries to the nearest energy minimum.

    pyannote boundaries can land mid-syllable. This snaps each boundary
    to the lowest-energy point within ±snap_ms, producing cleaner cuts
    that fall on natural speech pauses.
    """
    snap_samples = int(snap_ms * sr / 1000)
    frame_len = max(1, int(3 * sr / 1000))  # 3ms analysis frame
    refined = []

    for seg in segments:
        new_seg = seg.copy()
        for key in ("start", "end"):
            center = int(seg[key] * sr)
            lo = max(0, center - snap_samples)
            hi = min(len(audio), center + snap_samples)
            chunk = audio[lo:hi]
            if len(chunk) < frame_len * 3:
                continue
            # Compute short-time energy in the window
            n_frames = (len(chunk) - frame_len) // frame_len
            if n_frames < 1:
                continue
            energies = np.array([
                np.sqrt(np.mean(chunk[i * frame_len:(i + 1) * frame_len] ** 2) + 1e-12)
                for i in range(n_frames)
            ])
            min_idx = np.argmin(energies)
            refined_sample = lo + min_idx * frame_len + frame_len // 2
            new_seg[key] = refined_sample / sr

        if new_seg["end"] > new_seg["start"] + 0.01:
            refined.append(new_seg)

    return refined


def postprocess_segments(
    segments: list[dict],
    audio: np.ndarray | None = None,
    sr: int = 16000,
    min_segment_duration: float = 0.25,
    max_gap_merge: float = 0.3,
    min_island_duration: float = 0.15,
    snap_ms: int = 30,
) -> list[dict]:
    """Clean up raw diarization output for more accurate speaker separation.

    Five-pass algorithm:
      1. Merge same-speaker segments separated by short gaps
      2. Remove short "island" segments surrounded by another speaker
      3. Absorb remaining short segments into neighbors
      4. Final same-speaker merge pass
      5. Snap boundaries to energy minima (if audio provided)
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
                cleaned[-1]["end"] = max(cleaned[-1]["end"], seg["end"])
            else:
                cleaned.append(seg.copy())
        cleaned.append(merged[-1].copy())
    else:
        cleaned = merged

    # Pass 3: Absorb short segments into neighbors
    result = []
    for seg in cleaned:
        duration = seg["end"] - seg["start"]
        if duration >= min_segment_duration:
            result.append(seg)
        elif result:
            result[-1]["end"] = max(result[-1]["end"], seg["end"])

    if not result:
        return cleaned

    # Pass 4: Final same-speaker merge
    final = [result[0].copy()]
    for seg in result[1:]:
        prev = final[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            final.append(seg.copy())

    # Pass 5: Boundary refinement
    if audio is not None and snap_ms > 0:
        final = refine_boundaries(final, audio, sr, snap_ms)

    return final


# ---------------------------------------------------------------------------
# Stage 4: Channel-Informed Diarization Correction
# ---------------------------------------------------------------------------


def correct_segments_with_channels(
    segments: list[dict],
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    sr: int,
    speaker_a_label: str,
    speaker_b_label: str,
    min_duration_for_correction: float = 0.5,
    energy_ratio_threshold: float = 0.65,
) -> list[dict]:
    """Correct speaker labels using channel energy agreement.

    For each segment, check if the assigned speaker's primary channel has
    more energy than the other channel. If not, flip the label.

    Only corrects segments longer than min_duration_for_correction to avoid
    flipping short segments based on noisy energy estimates.
    """
    # First, establish which channel maps to which speaker
    # by majority vote across all segments
    a_on_ch_a = []
    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        if end <= start or end > len(channel_a):
            continue
        e_a = np.sqrt(np.mean(channel_a[start:end] ** 2) + 1e-10)
        e_b = np.sqrt(np.mean(channel_b[start:end] ** 2) + 1e-10)
        ratio = e_a / (e_a + e_b)
        if seg["speaker"] == speaker_a_label:
            a_on_ch_a.append(ratio)
        else:
            a_on_ch_a.append(1.0 - ratio)

    if not a_on_ch_a:
        return segments

    # If overall correlation is weak, don't correct (mics too similar)
    overall_agreement = np.mean(a_on_ch_a)
    if overall_agreement < 0.55:
        print(f"  [Correction] Channel agreement too weak ({overall_agreement:.2f}), skipping")
        return segments

    # Determine: speaker_a → channel_a or channel_b?
    a_prefers_ch_a = overall_agreement > 0.5

    corrected = []
    flip_count = 0
    for seg in segments:
        new_seg = seg.copy()
        duration = seg["end"] - seg["start"]
        if duration < min_duration_for_correction:
            corrected.append(new_seg)
            continue

        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        if end <= start or end > len(channel_a):
            corrected.append(new_seg)
            continue

        e_a = np.sqrt(np.mean(channel_a[start:end] ** 2) + 1e-10)
        e_b = np.sqrt(np.mean(channel_b[start:end] ** 2) + 1e-10)

        if a_prefers_ch_a:
            # Speaker A should be louder on channel A
            if seg["speaker"] == speaker_a_label:
                if e_b / (e_a + e_b) > energy_ratio_threshold:
                    new_seg["speaker"] = speaker_b_label
                    flip_count += 1
            else:
                if e_a / (e_a + e_b) > energy_ratio_threshold:
                    new_seg["speaker"] = speaker_a_label
                    flip_count += 1
        else:
            # Speaker A is louder on channel B (reversed mics)
            if seg["speaker"] == speaker_a_label:
                if e_a / (e_a + e_b) > energy_ratio_threshold:
                    new_seg["speaker"] = speaker_b_label
                    flip_count += 1
            else:
                if e_b / (e_a + e_b) > energy_ratio_threshold:
                    new_seg["speaker"] = speaker_a_label
                    flip_count += 1

        corrected.append(new_seg)

    if flip_count > 0:
        print(f"  [Correction] Flipped {flip_count}/{len(segments)} segments using channel energy")

    return corrected


# ---------------------------------------------------------------------------
# Overlap Detection
# ---------------------------------------------------------------------------


def find_overlaps(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split segments into non-overlapping and overlapping regions."""
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
# Stage 5: Speaker Embedding Extraction
# ---------------------------------------------------------------------------


def extract_speaker_embeddings(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    embedder,
    num_enrollment: int = 8,
    min_duration: float = 0.5,
) -> dict:
    """Extract duration-weighted centroid embedding for each speaker."""
    import torch

    speakers = sorted(set(s["speaker"] for s in segments))
    speaker_embeddings = {}

    for spk in speakers:
        spk_segs = [s for s in segments if s["speaker"] == spk]
        spk_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)
        eligible = [s for s in spk_segs if (s["end"] - s["start"]) >= min_duration]

        if not eligible:
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
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()
            stacked = torch.stack(embeddings)
            centroid = (stacked * weights_tensor.unsqueeze(1)).sum(dim=0)
            speaker_embeddings[spk] = centroid
        else:
            print(f"[WARN] No embeddings extracted for speaker {spk}")

    return speaker_embeddings


# ---------------------------------------------------------------------------
# Stage 6: Speaker Assignment
# ---------------------------------------------------------------------------


def assign_speakers(
    segments: list[dict],
    speaker_embeddings: dict,
    channel_a_audio: np.ndarray | None = None,
    channel_b_audio: np.ndarray | None = None,
    sr: int = 16000,
) -> tuple[str, str]:
    """Determine which diarization label maps to Speaker A vs Speaker B."""
    speakers = sorted(speaker_embeddings.keys())
    if len(speakers) < 2:
        return speakers[0] if speakers else "SPEAKER_00", "SPEAKER_01"

    first_speaker = None
    min_start = float("inf")
    for seg in segments:
        if seg["start"] < min_start:
            min_start = seg["start"]
            first_speaker = seg["speaker"]

    speaker_a_label = first_speaker
    speaker_b_label = [s for s in speakers if s != speaker_a_label][0]

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
    """Swap speaker labels if channel-energy correlation says they're reversed."""
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

    a_mean = np.mean(ch_a_ratio[speaker_a_label]) if ch_a_ratio[speaker_a_label] else 0.5
    b_mean = np.mean(ch_a_ratio[speaker_b_label]) if ch_a_ratio[speaker_b_label] else 0.5

    if a_mean < 0.4 and b_mean > 0.6:
        print(f"  [Channel validation] Swapping labels: "
              f"A_ch_a={a_mean:.2f}, B_ch_a={b_mean:.2f}")
        return speaker_b_label, speaker_a_label

    return speaker_a_label, speaker_b_label


# ---------------------------------------------------------------------------
# Stage 7: Cross-Channel Leakage Estimation
# ---------------------------------------------------------------------------


def estimate_leakage(
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
) -> tuple[float, float]:
    """Estimate cross-channel leakage ratios from non-overlap segments.

    During non-overlap Speaker A segments:
      channel_a ≈ Speaker A (strong)
      channel_b ≈ Speaker A * leak_a_to_b (weak, cross-talk)

    Returns (leak_a_to_b, leak_b_to_a) — values between 0 and 0.95.
    """
    ratios_a2b = []
    ratios_b2a = []

    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        if end <= start or end > len(channel_a):
            continue

        e_a = np.sqrt(np.mean(channel_a[start:end] ** 2) + 1e-10)
        e_b = np.sqrt(np.mean(channel_b[start:end] ** 2) + 1e-10)

        if seg["speaker"] == speaker_a_label and e_a > 1e-6:
            ratios_a2b.append(e_b / e_a)
        elif seg["speaker"] == speaker_b_label and e_b > 1e-6:
            ratios_b2a.append(e_a / e_b)

    leak_a2b = float(np.clip(np.median(ratios_a2b), 0.0, 0.95)) if ratios_a2b else 0.5
    leak_b2a = float(np.clip(np.median(ratios_b2a), 0.0, 0.95)) if ratios_b2a else 0.5

    return leak_a2b, leak_b2a


# ---------------------------------------------------------------------------
# Stage 8: Speaker Spectral Profiles (for Wiener overlap separation)
# ---------------------------------------------------------------------------


def compute_spectral_profiles(
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    n_fft: int = 1024,
    hop_length: int = 256,
    max_segments: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Build average magnitude spectral profiles per speaker from clean segments.

    Uses each speaker's primary channel during non-overlap regions to get
    a clean frequency fingerprint. These profiles drive Wiener masking.

    Returns (profile_a, profile_b) — each shape (n_fft//2 + 1,).
    """
    import librosa

    freq_bins = n_fft // 2 + 1

    def _profile_from_segments(audio, segs):
        specs = []
        used = 0
        for seg in sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True):
            if used >= max_segments:
                break
            start = int(seg["start"] * sr)
            end = int(seg["end"] * sr)
            chunk = audio[start:end]
            if len(chunk) < n_fft:
                continue
            S = np.abs(librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length))
            specs.append(np.mean(S, axis=1))
            used += 1
        if specs:
            return np.mean(specs, axis=0).astype(np.float32)
        return np.ones(freq_bins, dtype=np.float32)

    a_segs = [s for s in non_overlap_segments if s["speaker"] == speaker_a_label]
    b_segs = [s for s in non_overlap_segments if s["speaker"] == speaker_b_label]

    profile_a = _profile_from_segments(channel_a, a_segs)
    profile_b = _profile_from_segments(channel_b, b_segs)

    return profile_a, profile_b


# ---------------------------------------------------------------------------
# Stage 9: Separation helpers
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
    """Apply a chunk with cosine fades into the track buffer (overlap-add)."""
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


def leakage_suppress_chunk(
    primary: np.ndarray,
    secondary: np.ndarray,
    leak_ratio: float,
    alpha: float = 0.7,
    floor: float = 0.08,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Remove cross-channel leakage via spectral subtraction.

    primary   = Speaker A signal + noise
    secondary = Speaker A leakage (attenuated) + Speaker B signal
    leak_ratio = estimated energy ratio of leakage

    We subtract the estimated leakage in the spectral domain with a floor
    to prevent musical noise artifacts:
      |S_clean| = max(|S_primary| - alpha * leak * |S_secondary|, floor * |S_primary|)
    """
    import librosa

    if len(primary) < n_fft or leak_ratio < 0.05:
        return primary

    S_prim = librosa.stft(primary, n_fft=n_fft, hop_length=hop_length)
    S_sec = librosa.stft(secondary, n_fft=n_fft, hop_length=hop_length)

    mag_prim = np.abs(S_prim)
    mag_sec = np.abs(S_sec)
    phase_prim = np.angle(S_prim)

    # Spectral subtraction with floor
    mag_clean = np.maximum(
        mag_prim - alpha * leak_ratio * mag_sec,
        floor * mag_prim,
    )

    S_clean = mag_clean * np.exp(1j * phase_prim)
    result = librosa.istft(S_clean, hop_length=hop_length, length=len(primary))
    return result.astype(np.float32)


def wiener_separate_overlap(
    channel_a_chunk: np.ndarray,
    channel_b_chunk: np.ndarray,
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate an overlap region using Wiener filtering with speaker profiles.

    Uses the pre-computed spectral profiles to create soft frequency masks:
      mask_a = |profile_a|² / (|profile_a|² + |profile_b|² + eps)

    Applied to the mixed signal to extract each speaker.
    """
    import librosa

    mix = (channel_a_chunk + channel_b_chunk) / 2.0
    chunk_len = len(mix)

    if chunk_len < n_fft:
        return channel_a_chunk.copy(), channel_b_chunk.copy()

    S_mix = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)

    # Wiener masks from speaker profiles
    pa2 = (profile_a ** 2)[:, np.newaxis]
    pb2 = (profile_b ** 2)[:, np.newaxis]
    denom = pa2 + pb2 + 1e-10

    mask_a = pa2 / denom
    mask_b = pb2 / denom

    S_a = S_mix * mask_a
    S_b = S_mix * mask_b

    src_a = librosa.istft(S_a, hop_length=hop_length, length=chunk_len)
    src_b = librosa.istft(S_b, hop_length=hop_length, length=chunk_len)

    return src_a.astype(np.float32), src_b.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 10: VAD Gating (post-separation cleanup)
# ---------------------------------------------------------------------------


def vad_gate_track(
    track: np.ndarray,
    sr: int,
    threshold_db: float = -40,
    attack_ms: int = 5,
    release_ms: int = 80,
    frame_ms: int = 10,
) -> np.ndarray:
    """Zero out non-speech regions using energy-based VAD with smooth gating.

    After separation, each track may still have faint residual bleed from the
    other speaker in "silent" portions. This gate silences those regions.

    Uses attack/release smoothing to avoid choppy gating artifacts:
    - attack: fast fade-in when speech starts (5ms)
    - release: slow fade-out after speech ends (80ms) to preserve natural tails
    """
    frame_len = max(1, int(frame_ms * sr / 1000))
    n_frames = max(1, len(track) // frame_len)

    # Compute per-frame RMS energy
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(track))
        energies[i] = np.sqrt(np.mean(track[start:end] ** 2) + 1e-12)

    # Threshold: relative to track's own peak energy
    peak_energy = np.max(energies) if len(energies) > 0 else 1e-6
    if peak_energy < 1e-8:
        return track
    threshold = peak_energy * (10 ** (threshold_db / 20))
    is_speech = energies > threshold

    # Attack/release envelope smoothing
    attack_frames = max(1, int(attack_ms / frame_ms))
    release_frames = max(1, int(release_ms / frame_ms))

    envelope = np.zeros(n_frames, dtype=np.float32)
    current = 0.0
    for i in range(n_frames):
        if is_speech[i]:
            # Attack: fast rise
            current = min(1.0, current + 1.0 / attack_frames)
        else:
            # Release: slow decay
            current = max(0.0, current - 1.0 / release_frames)
        envelope[i] = current

    # Expand envelope to sample-level
    frame_centers = np.arange(n_frames) * frame_len + frame_len // 2
    sample_indices = np.arange(len(track))
    gate = np.interp(sample_indices, frame_centers, envelope).astype(np.float32)

    return track * gate


# ---------------------------------------------------------------------------
# Stage 9a: Channel-Aware Separation (Pipeline B) — with leakage + Wiener
# ---------------------------------------------------------------------------


def channel_aware_separate(
    channel_a_audio: np.ndarray,
    channel_b_audio: np.ndarray,
    sr: int,
    non_overlap_segments: list[dict],
    overlap_regions: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    leak_a2b: float = 0.5,
    leak_b2a: float = 0.5,
    profile_a: np.ndarray | None = None,
    profile_b: np.ndarray | None = None,
    fade_ms: int = 20,
    leakage_alpha: float = 0.7,
    leakage_floor: float = 0.08,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> dict:
    """Build per-speaker tracks with leakage suppression and Wiener overlap separation.

    Non-overlaps: extract from primary channel + spectral leakage suppression
    to remove the other speaker's bleed.

    Overlaps: Wiener filtering using speaker spectral profiles to separate
    the speakers in the frequency domain.
    """
    total_samples = len(channel_a_audio)
    fade_samples = int(fade_ms * sr / 1000)

    track_a = np.zeros(total_samples, dtype=np.float32)
    track_b = np.zeros(total_samples, dtype=np.float32)

    label_to_role = {speaker_a_label: "speaker_a", speaker_b_label: "speaker_b"}

    # Non-overlaps: primary channel + leakage suppression
    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = min(int(seg["end"] * sr), total_samples)
        if end <= start:
            continue
        role = label_to_role.get(seg["speaker"])

        if role == "speaker_a":
            primary = channel_a_audio[start:end].copy()
            secondary = channel_b_audio[start:end].copy()
            cleaned = leakage_suppress_chunk(
                primary, secondary, leak_b2a, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            _apply_faded_chunk(track_a, start, end, cleaned, fade_samples)
        elif role == "speaker_b":
            primary = channel_b_audio[start:end].copy()
            secondary = channel_a_audio[start:end].copy()
            cleaned = leakage_suppress_chunk(
                primary, secondary, leak_a2b, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            _apply_faded_chunk(track_b, start, end, cleaned, fade_samples)

    # Overlaps: per-channel leakage suppression (avoids mixing channels
    # which doubles cross-talk and creates echo artifacts)
    for region in overlap_regions:
        start = int(region["start"] * sr)
        end = min(int(region["end"] * sr), total_samples)
        if end - start < int(0.05 * sr):
            continue

        # Speaker A: primary = ch_a, suppress Speaker B leakage
        primary_a = channel_a_audio[start:end].copy()
        secondary_a = channel_b_audio[start:end].copy()
        cleaned_a = leakage_suppress_chunk(
            primary_a, secondary_a, leak_b2a, leakage_alpha, leakage_floor, n_fft, hop_length
        )
        _apply_faded_chunk(track_a, start, end, cleaned_a, fade_samples)

        # Speaker B: primary = ch_b, suppress Speaker A leakage
        primary_b = channel_b_audio[start:end].copy()
        secondary_b = channel_a_audio[start:end].copy()
        cleaned_b = leakage_suppress_chunk(
            primary_b, secondary_b, leak_a2b, leakage_alpha, leakage_floor, n_fft, hop_length
        )
        _apply_faded_chunk(track_b, start, end, cleaned_b, fade_samples)

    return {"speaker_a": track_a, "speaker_b": track_b}


# ---------------------------------------------------------------------------
# Stage 9b: SepFormer Separation (Pipeline C) — with leakage + Wiener + SepFormer
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
    """Run SepFormer on an overlap chunk and assign via embedding matching."""
    import torch
    import torch.nn.functional as F

    min_samples = int(0.5 * sr)
    original_len = len(mixed_chunk)

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

    src1 = src1[context_len:context_len + original_len]
    src2 = src2[context_len:context_len + original_len]

    a_emb = speaker_embeddings.get(speaker_a_label)
    b_emb = speaker_embeddings.get(speaker_b_label)

    if a_emb is None or b_emb is None:
        return src1, src2

    if len(src1) < sr * 0.1 or len(src2) < sr * 0.1:
        return src1, src2

    emb1 = embedder.encode_batch(torch.from_numpy(src1).unsqueeze(0).float()).squeeze().cpu()
    emb2 = embedder.encode_batch(torch.from_numpy(src2).unsqueeze(0).float()).squeeze().cpu()

    sim_1a = F.cosine_similarity(emb1.unsqueeze(0), a_emb.unsqueeze(0)).item()
    sim_2a = F.cosine_similarity(emb2.unsqueeze(0), a_emb.unsqueeze(0)).item()
    sim_1b = F.cosine_similarity(emb1.unsqueeze(0), b_emb.unsqueeze(0)).item()
    sim_2b = F.cosine_similarity(emb2.unsqueeze(0), b_emb.unsqueeze(0)).item()

    score_normal = sim_1a + sim_2b
    score_swapped = sim_2a + sim_1b

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
    leak_a2b: float = 0.5,
    leak_b2a: float = 0.5,
    fade_ms: int = 20,
    context_s: float = 0.25,
    leakage_alpha: float = 0.7,
    leakage_floor: float = 0.08,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> dict:
    """Hybrid: leakage-suppressed non-overlaps + SepFormer for overlaps."""
    total_samples = len(channel_a_audio)
    fade_samples = int(fade_ms * sr / 1000)
    context_samples = int(context_s * sr)

    track_a = np.zeros(total_samples, dtype=np.float32)
    track_b = np.zeros(total_samples, dtype=np.float32)

    label_to_role = {speaker_a_label: "speaker_a", speaker_b_label: "speaker_b"}

    # Non-overlaps: primary channel + leakage suppression
    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = min(int(seg["end"] * sr), total_samples)
        if end <= start:
            continue
        role = label_to_role.get(seg["speaker"])

        if role == "speaker_a":
            primary = channel_a_audio[start:end].copy()
            secondary = channel_b_audio[start:end].copy()
            cleaned = leakage_suppress_chunk(
                primary, secondary, leak_b2a, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            _apply_faded_chunk(track_a, start, end, cleaned, fade_samples)
        elif role == "speaker_b":
            primary = channel_b_audio[start:end].copy()
            secondary = channel_a_audio[start:end].copy()
            cleaned = leakage_suppress_chunk(
                primary, secondary, leak_a2b, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            _apply_faded_chunk(track_b, start, end, cleaned, fade_samples)

    # Overlaps: SepFormer + embedding matching
    for region in overlap_regions:
        start = int(region["start"] * sr)
        end = min(int(region["end"] * sr), total_samples)
        if end - start < int(0.05 * sr):
            continue

        mixed_chunk = mixed_audio[start:end]
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
            # Fallback: leakage suppression from primary channels
            cleaned_a = leakage_suppress_chunk(
                channel_a_audio[start:end].copy(), channel_b_audio[start:end].copy(),
                leak_b2a, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            cleaned_b = leakage_suppress_chunk(
                channel_b_audio[start:end].copy(), channel_a_audio[start:end].copy(),
                leak_a2b, leakage_alpha, leakage_floor, n_fft, hop_length
            )
            _apply_faded_chunk(track_a, start, end, cleaned_a, fade_samples)
            _apply_faded_chunk(track_b, start, end, cleaned_b, fade_samples)

    return {"speaker_a": track_a, "speaker_b": track_b}


# ---------------------------------------------------------------------------
# Stage 11: Final Post-processing
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
# Shared Stages
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
    snap_ms: int = 30,
) -> dict:
    """Run Stages 0-8, return all intermediate state for separation."""
    print("[Stage 0] Loading audio channels...")
    ch_a_audio, sr = load_and_preprocess(channel_a_bytes)
    ch_b_audio, _ = load_and_preprocess(channel_b_bytes)
    mixed_audio, _ = load_and_preprocess(mixed_bytes)
    print(f"  Duration: {len(mixed_audio)/sr:.1f}s @ {sr} Hz")

    # Also denoise individual channels for better separation later
    if denoise_enabled:
        print("[Stage 1] Denoising signals...")
        mixed_clean = denoise_audio(mixed_audio, sr, denoise_prop)
        ch_a_clean = denoise_audio(ch_a_audio, sr, denoise_prop)
        ch_b_clean = denoise_audio(ch_b_audio, sr, denoise_prop)
    else:
        mixed_clean = mixed_audio
        ch_a_clean = ch_a_audio
        ch_b_clean = ch_b_audio

    print("[Stage 2] Diarizing (finding who spoke when)...")
    raw_segments = diarize(mixed_clean, sr, diarizer, num_speakers)
    speakers_found = set(s["speaker"] for s in raw_segments)
    print(f"  Found {len(speakers_found)} speakers, {len(raw_segments)} raw segments")

    print("[Stage 3] Post-processing diarization (merge + boundary refinement)...")
    segments = postprocess_segments(
        raw_segments, mixed_clean, sr,
        min_segment_duration, max_gap_merge, min_island_duration, snap_ms,
    )
    print(f"  Cleaned: {len(raw_segments)} -> {len(segments)} segments "
          f"(removed {len(raw_segments) - len(segments)} fragments)")

    print("[Stage 4] Extracting speaker embeddings (duration-weighted)...")
    # Use non-overlap segments for embedding extraction (computed after initial overlap detection)
    non_overlaps_raw, overlaps_raw = find_overlaps(segments)
    speaker_embeddings = extract_speaker_embeddings(
        mixed_clean, sr, non_overlaps_raw, embedder, num_enrollment, min_enrollment_duration
    )
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

    # Re-compute overlaps after correction
    non_overlaps, overlaps = find_overlaps(segments)
    total_dur = len(mixed_audio) / sr
    overlap_dur = sum(r["end"] - r["start"] for r in overlaps)
    overlap_pct = (overlap_dur / total_dur * 100) if total_dur > 0 else 0
    print(f"  Non-overlap: {len(non_overlaps)} segments, "
          f"Overlap: {len(overlaps)} regions ({overlap_dur:.1f}s / {overlap_pct:.1f}%)")

    print("[Stage 7] Estimating cross-channel leakage...")
    leak_a2b, leak_b2a = estimate_leakage(
        ch_a_clean, ch_b_clean, sr, non_overlaps, speaker_a_label, speaker_b_label,
    )
    print(f"  Leakage: A->B = {leak_a2b:.3f}, B->A = {leak_b2a:.3f}")

    print("[Stage 8] Building speaker spectral profiles for Wiener masking...")
    profile_a, profile_b = compute_spectral_profiles(
        ch_a_clean, ch_b_clean, sr, non_overlaps, speaker_a_label, speaker_b_label,
    )
    print(f"  Profiles: A shape={profile_a.shape}, B shape={profile_b.shape}")

    return {
        "ch_a_audio": ch_a_clean,
        "ch_b_audio": ch_b_clean,
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
        "leak_a2b": leak_a2b,
        "leak_b2a": leak_b2a,
        "profile_a": profile_a,
        "profile_b": profile_b,
    }


# ---------------------------------------------------------------------------
# Pipeline B: Channel-Aware + Leakage Suppression + Wiener Overlaps + VAD
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
    """Pipeline B: channel-aware separation with leakage suppression + Wiener overlaps.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline B: Channel-Aware + Wiener] {session_name}")
    print(f"{'='*60}")

    t0 = time.time()
    state = _run_shared_stages(
        channel_a_bytes, channel_b_bytes, mixed_bytes, session_name,
        diarizer, embedder, denoise_enabled, denoise_prop,
        num_speakers, num_enrollment, min_enrollment_duration,
    )

    print("[Stage 9] Channel-aware separation with leakage suppression + Wiener...")
    tracks = channel_aware_separate(
        state["ch_a_audio"], state["ch_b_audio"], state["sr"],
        state["non_overlaps"], state["overlaps"],
        state["speaker_a_label"], state["speaker_b_label"],
        state["leak_a2b"], state["leak_b2a"],
        state["profile_a"], state["profile_b"],
        fade_ms,
    )

    print("[Stage 10] VAD gating (silencing non-speech residuals)...")
    for key in ["speaker_a", "speaker_b"]:
        tracks[key] = vad_gate_track(tracks[key], state["sr"])

    print("[Stage 11] Normalizing and encoding...")
    elapsed = round(time.time() - t0, 1)
    result = {
        "speaker_a": postprocess_track(tracks["speaker_a"], state["sr"], peak_db),
        "speaker_b": postprocess_track(tracks["speaker_b"], state["sr"], peak_db),
        "metadata": {
            "elapsed_s": elapsed,
            "segments": state["num_segments"],
            "raw_segments": state["num_raw_segments"],
            "overlap_pct": state["overlap_pct"],
            "leak_a2b": state["leak_a2b"],
            "leak_b2a": state["leak_b2a"],
        },
    }

    for key in ["speaker_a", "speaker_b"]:
        print(f"  Output '{key}': {len(result[key])/1024:.0f} KB")
    print(f"[Pipeline B DONE] {elapsed}s\n")
    return result


# ---------------------------------------------------------------------------
# Pipeline C: SepFormer + Leakage Suppression + VAD
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
    """Pipeline C: SepFormer overlaps + leakage-suppressed non-overlaps.

    Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline C: SepFormer + Leakage] {session_name}")
    print(f"{'='*60}")

    t0 = time.time()
    state = _run_shared_stages(
        channel_a_bytes, channel_b_bytes, mixed_bytes, session_name,
        diarizer, embedder, denoise_enabled, denoise_prop,
        num_speakers, num_enrollment, min_enrollment_duration,
    )

    print("[Stage 9] SepFormer separation with leakage suppression...")
    tracks = sepformer_separate(
        state["ch_a_audio"], state["ch_b_audio"], state["mixed_audio"], state["sr"],
        state["non_overlaps"], state["overlaps"],
        state["speaker_a_label"], state["speaker_b_label"],
        separator, embedder, state["speaker_embeddings"],
        state["leak_a2b"], state["leak_b2a"],
        fade_ms, context_s,
    )

    print("[Stage 10] VAD gating (silencing non-speech residuals)...")
    for key in ["speaker_a", "speaker_b"]:
        tracks[key] = vad_gate_track(tracks[key], state["sr"])

    print("[Stage 11] Normalizing and encoding...")
    elapsed = round(time.time() - t0, 1)
    result = {
        "speaker_a": postprocess_track(tracks["speaker_a"], state["sr"], peak_db),
        "speaker_b": postprocess_track(tracks["speaker_b"], state["sr"], peak_db),
        "metadata": {
            "elapsed_s": elapsed,
            "segments": state["num_segments"],
            "raw_segments": state["num_raw_segments"],
            "overlap_pct": state["overlap_pct"],
            "leak_a2b": state["leak_a2b"],
            "leak_b2a": state["leak_b2a"],
        },
    }

    for key in ["speaker_a", "speaker_b"]:
        print(f"  Output '{key}': {len(result[key])/1024:.0f} KB")
    print(f"[Pipeline C DONE] {elapsed}s\n")
    return result
