"""Energy-based speaker separation — runs LOCALLY, no GPU needed.

Compares per-frame RMS energy between the two mic channels to determine
which speaker is active, then extracts from the appropriate mic.
"""

import io

import numpy as np
import soundfile as sf


def _rms_energy(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Compute per-frame RMS energy."""
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return energy


def _cosine_fade(length: int, fade_in: bool = True) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, np.pi / 2, length)
    return np.sin(t).astype(np.float32) if fade_in else np.cos(t).astype(np.float32)


def run_energy_pipeline(
    channel_a_bytes: bytes,
    channel_b_bytes: bytes,
    sr: int = 16000,
    frame_ms: int = 25,
    hop_ms: int = 10,
    fade_ms: int = 15,
    peak_db: float = -1.0,
) -> dict:
    """Separate speakers using per-frame energy comparison between mics.

    Returns: {"speaker_a": wav_bytes, "speaker_b": wav_bytes}
    """
    ch_a_audio, actual_sr = sf.read(io.BytesIO(channel_a_bytes), dtype="float32")
    ch_b_audio, _ = sf.read(io.BytesIO(channel_b_bytes), dtype="float32")

    if ch_a_audio.ndim > 1:
        ch_a_audio = np.mean(ch_a_audio, axis=1)
    if ch_b_audio.ndim > 1:
        ch_b_audio = np.mean(ch_b_audio, axis=1)

    # Use the actual sample rate from the file
    sr = actual_sr

    frame_len = int(frame_ms * sr / 1000)
    hop_len = int(hop_ms * sr / 1000)
    fade_samples = int(fade_ms * sr / 1000)

    # Pad to same length
    max_len = max(len(ch_a_audio), len(ch_b_audio))
    if len(ch_a_audio) < max_len:
        ch_a_audio = np.pad(ch_a_audio, (0, max_len - len(ch_a_audio)))
    if len(ch_b_audio) < max_len:
        ch_b_audio = np.pad(ch_b_audio, (0, max_len - len(ch_b_audio)))

    a_energy = _rms_energy(ch_a_audio, frame_len, hop_len)
    b_energy = _rms_energy(ch_b_audio, frame_len, hop_len)

    # Soft mask: ratio of energies (how much louder each mic is)
    total_energy = a_energy + b_energy + 1e-10
    a_ratio = a_energy / total_energy
    b_ratio = b_energy / total_energy

    # Expand frame-level masks to sample-level with interpolation
    n_samples = max_len
    frame_times = np.arange(len(a_ratio)) * hop_len + frame_len // 2
    sample_indices = np.arange(n_samples)
    a_mask = np.interp(sample_indices, frame_times, a_ratio).astype(np.float32)
    b_mask = np.interp(sample_indices, frame_times, b_ratio).astype(np.float32)

    # Sharpen masks slightly (push toward 0/1 while keeping softness)
    a_mask = np.clip(a_mask * 1.5 - 0.25, 0, 1)
    b_mask = np.clip(b_mask * 1.5 - 0.25, 0, 1)

    # Apply masks: each track gets its own mic weighted by the mask
    track_a = ch_a_audio[:n_samples] * a_mask
    track_b = ch_b_audio[:n_samples] * b_mask

    # Apply fades at silence/speech transitions to reduce artifacts
    for track in [track_a, track_b]:
        _apply_transition_fades(track, fade_samples)

    # Normalize
    result = {}
    for key, track in [("speaker_a", track_a), ("speaker_b", track_b)]:
        peak = np.max(np.abs(track))
        if peak > 0:
            target = 10 ** (peak_db / 20)
            track = track * (target / peak)
        track = np.clip(track, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, track, int(sr), subtype="PCM_16", format="WAV")
        result[key] = buf.getvalue()

    return result


def _apply_transition_fades(audio: np.ndarray, fade_len: int, threshold: float = 0.01):
    """Apply cosine fades at transitions from silence to speech and back."""
    if fade_len <= 0:
        return

    is_active = np.abs(audio) > threshold
    # Find transitions
    diff = np.diff(is_active.astype(np.int8))
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1

    for onset in onsets:
        end = min(onset + fade_len, len(audio))
        actual_len = end - onset
        if actual_len > 0:
            audio[onset:end] *= _cosine_fade(actual_len, fade_in=True)

    for offset in offsets:
        start = max(offset - fade_len, 0)
        actual_len = offset - start
        if actual_len > 0:
            audio[start:offset] *= _cosine_fade(actual_len, fade_in=False)
