"""Local audio utilities — loading, format detection, basic I/O.

Runs on the Mac side only. No GPU dependencies required.
"""

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def find_session_folder(input_dir: Path, session_num: int) -> Optional[Path]:
    """Flexibly locate a session folder regardless of naming convention.

    Handles: Session_001, Session 1, Session1, session_1, 001, etc.
    """
    candidates = [
        f"Session_{session_num:03d}",
        f"Session_{session_num}",
        f"Session {session_num}",
        f"Session{session_num}",
        f"session_{session_num:03d}",
        f"session_{session_num}",
        f"{session_num:03d}",
        f"{session_num}",
    ]
    for name in candidates:
        path = input_dir / name
        if path.is_dir():
            return path

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        match = re.search(r"(\d+)", child.name)
        if match and int(match.group(1)) == session_num:
            return child

    return None


def discover_sessions(input_dir: Path) -> list[int]:
    """Return sorted list of session numbers found in input_dir."""
    sessions = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        match = re.search(r"(\d+)", child.name)
        if match:
            sessions.append(int(match.group(1)))
    return sorted(set(sessions))


def _read_wav_mono_16k(wav_path: Path) -> np.ndarray:
    """Read a WAV file, downmix to mono, resample to 16 kHz, normalize."""
    data, sr = sf.read(wav_path, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    return data


def _encode_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode float32 audio array to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _classify_mic(filename: str) -> str:
    """Determine mic channel from filename. Returns 'a', 'b', or 'unknown'.

    Supports naming conventions:
    - doctor/doc/speaker_a/mic_a/mic1/ch1/channel_1 → channel A
    - patient/pat/speaker_b/mic_b/mic2/ch2/channel_2 → channel B
    """
    lower = filename.lower()
    # Channel A patterns (first mic / doctor / speaker A)
    if any(p in lower for p in ["doctor", "doc", "speaker_a", "mic_a", "mic1", "mic_1", "ch1", "ch_1", "channel_1", "channel_a"]):
        return "a"
    # Channel B patterns (second mic / patient / speaker B)
    if any(p in lower for p in ["patient", "pat", "speaker_b", "mic_b", "mic2", "mic_2", "ch2", "ch_2", "channel_2", "channel_b"]):
        return "b"
    return "unknown"


def load_session_audio_separate(session_dir: Path) -> tuple[dict, dict]:
    """Load session WAVs keeping channels separate, identified by mic role.

    Returns:
        (channels, info) where channels is:
          {"channel_a_bytes": bytes, "channel_b_bytes": bytes, "mixed_bytes": bytes}
        and info has file metadata.

    If mic roles can't be identified from filenames, falls back to ordering
    (first file = channel A, second = channel B).
    """
    wav_files = sorted(session_dir.glob("*.wav")) + sorted(session_dir.glob("*.WAV"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {session_dir}")

    seen = set()
    unique_files = []
    for f in wav_files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique_files.append(f)
    wav_files = unique_files

    # Classify files by mic channel
    classified: dict[str, list[tuple[Path, np.ndarray]]] = {
        "a": [], "b": [], "unknown": [],
    }
    file_info = []

    for wav_path in wav_files:
        data = _read_wav_mono_16k(wav_path)
        role = _classify_mic(wav_path.name)
        classified[role].append((wav_path, data))
        file_info.append({
            "name": wav_path.name,
            "role": role,
            "duration_s": len(data) / 16000,
            "size_bytes": wav_path.stat().st_size,
        })

    # Resolve channels: prefer classified, fall back to positional
    if classified["a"] and classified["b"]:
        ch_a_audio = classified["a"][0][1]
        ch_b_audio = classified["b"][0][1]
    elif len(wav_files) >= 2:
        arrays = [_read_wav_mono_16k(f) for f in wav_files[:2]]
        ch_a_audio, ch_b_audio = arrays[0], arrays[1]
    elif len(wav_files) == 1:
        only = _read_wav_mono_16k(wav_files[0])
        ch_a_audio = only
        ch_b_audio = only
    else:
        raise FileNotFoundError(f"No WAV files found in {session_dir}")

    # Pad to same length
    max_len = max(len(ch_a_audio), len(ch_b_audio))
    if len(ch_a_audio) < max_len:
        ch_a_audio = np.pad(ch_a_audio, (0, max_len - len(ch_a_audio)))
    if len(ch_b_audio) < max_len:
        ch_b_audio = np.pad(ch_b_audio, (0, max_len - len(ch_b_audio)))

    # Create mix for diarization
    mixed = (ch_a_audio + ch_b_audio) / 2.0
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak

    channels = {
        "channel_a_bytes": _encode_wav_bytes(ch_a_audio),
        "channel_b_bytes": _encode_wav_bytes(ch_b_audio),
        "mixed_bytes": _encode_wav_bytes(mixed),
    }

    info = {
        "files": file_info,
        "num_files": len(wav_files),
        "duration_s": max_len / 16000,
        "total_bytes": sum(f["size_bytes"] for f in file_info),
    }

    return channels, info


def load_session_audio(session_dir: Path) -> tuple[bytes, dict]:
    """Legacy: load all WAVs into a single mixed mono signal."""
    channels, info = load_session_audio_separate(session_dir)
    return channels["mixed_bytes"], info


def save_wav_bytes(wav_bytes: bytes, output_path: Path) -> None:
    """Write raw WAV bytes to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)
