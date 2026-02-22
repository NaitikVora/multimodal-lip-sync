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

    # Regex fallback: any folder whose name contains the session number
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


def load_session_audio(session_dir: Path) -> tuple[bytes, dict]:
    """Load all WAV files in a session folder into a single mixed mono signal.

    Returns:
        (wav_bytes, info_dict) where wav_bytes is 16kHz mono 16-bit PCM WAV
        and info_dict has metadata about what was loaded.
    """
    wav_files = sorted(session_dir.glob("*.wav")) + sorted(session_dir.glob("*.WAV"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {session_dir}")

    # De-duplicate (case-insensitive filesystems)
    seen = set()
    unique_files = []
    for f in wav_files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique_files.append(f)
    wav_files = unique_files

    arrays = []
    file_info = []

    for wav_path in wav_files:
        data, sr = sf.read(wav_path, dtype="float32")

        # Downmix to mono if multi-channel
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000

        arrays.append(data)
        file_info.append({
            "name": wav_path.name,
            "original_sr": sr,
            "duration_s": len(data) / 16000,
            "size_bytes": wav_path.stat().st_size,
        })

    # Mix all arrays to a single mono signal (pad shorter ones with silence)
    max_len = max(len(a) for a in arrays)
    mixed = np.zeros(max_len, dtype=np.float32)
    for a in arrays:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[: len(a)] = a
        mixed += padded

    # Average to avoid clipping from summation
    if len(arrays) > 1:
        mixed /= len(arrays)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak

    # Encode as 16-bit PCM WAV bytes
    buf = io.BytesIO()
    sf.write(buf, mixed, 16000, subtype="PCM_16", format="WAV")
    wav_bytes = buf.getvalue()

    info = {
        "files": file_info,
        "num_files": len(wav_files),
        "mixed_duration_s": max_len / 16000,
        "total_bytes": sum(f["size_bytes"] for f in file_info),
    }

    return wav_bytes, info


def save_wav_bytes(wav_bytes: bytes, output_path: Path) -> None:
    """Write raw WAV bytes to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)
