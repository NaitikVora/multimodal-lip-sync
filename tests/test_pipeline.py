"""Tests for the speaker separation pipeline.

These tests validate local utilities and pipeline logic without requiring
a GPU or Modal deployment. GPU-dependent tests are marked with pytest.mark.skip.
"""

import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_wav_bytes(
    duration_s: float = 2.0,
    sr: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
) -> bytes:
    """Generate a synthetic WAV file as bytes."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    if channels > 1:
        signal = np.column_stack([signal] * channels)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def make_session_dir(tmp_path: Path, num_files: int = 2) -> Path:
    """Create a temporary session directory with synthetic WAV files."""
    session_dir = tmp_path / "Session_001"
    session_dir.mkdir()

    freqs = [440.0, 880.0]
    for i in range(num_files):
        name = f"Mic_{i + 1}.wav"
        wav_bytes = make_wav_bytes(freq=freqs[i % len(freqs)])
        (session_dir / name).write_bytes(wav_bytes)

    return session_dir


# ---------------------------------------------------------------------------
# Tests — audio_utils (local)
# ---------------------------------------------------------------------------


class TestAudioUtils:
    def test_find_session_folder(self, tmp_path):
        from src.audio_utils import find_session_folder

        (tmp_path / "Session_001").mkdir()
        (tmp_path / "Session_002").mkdir()

        assert find_session_folder(tmp_path, 1).name == "Session_001"
        assert find_session_folder(tmp_path, 2).name == "Session_002"
        assert find_session_folder(tmp_path, 99) is None

    def test_find_session_folder_flexible(self, tmp_path):
        from src.audio_utils import find_session_folder

        (tmp_path / "Session 3").mkdir()
        assert find_session_folder(tmp_path, 3).name == "Session 3"

    def test_discover_sessions(self, tmp_path):
        from src.audio_utils import discover_sessions

        for i in [1, 3, 10]:
            (tmp_path / f"Session_{i:03d}").mkdir()

        sessions = discover_sessions(tmp_path)
        assert sessions == [1, 3, 10]

    def test_load_session_audio_single_file(self, tmp_path):
        from src.audio_utils import load_session_audio

        session_dir = make_session_dir(tmp_path, num_files=1)
        wav_bytes, info = load_session_audio(session_dir)

        assert isinstance(wav_bytes, bytes)
        assert info["num_files"] == 1
        assert info["duration_s"] > 0

        # Verify it's valid WAV
        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 16000
        assert data.ndim == 1  # mono

    def test_load_session_audio_multiple_files(self, tmp_path):
        from src.audio_utils import load_session_audio

        session_dir = make_session_dir(tmp_path, num_files=2)
        wav_bytes, info = load_session_audio(session_dir)

        assert info["num_files"] == 2
        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 16000

    def test_load_session_audio_stereo(self, tmp_path):
        from src.audio_utils import load_session_audio

        session_dir = tmp_path / "Session_001"
        session_dir.mkdir()
        wav_bytes = make_wav_bytes(channels=2)
        (session_dir / "stereo.wav").write_bytes(wav_bytes)

        result_bytes, info = load_session_audio(session_dir)
        data, sr = sf.read(io.BytesIO(result_bytes))
        assert data.ndim == 1  # downmixed to mono

    def test_save_wav_bytes(self, tmp_path):
        from src.audio_utils import save_wav_bytes

        wav_bytes = make_wav_bytes()
        out_path = tmp_path / "sub" / "out.wav"
        save_wav_bytes(wav_bytes, out_path)

        assert out_path.exists()
        assert out_path.read_bytes() == wav_bytes


# ---------------------------------------------------------------------------
# Tests — pipeline functions (no GPU needed)
# ---------------------------------------------------------------------------


class TestPipelineStages:
    def test_load_and_preprocess(self):
        from src.pipeline import load_and_preprocess

        wav_bytes = make_wav_bytes(duration_s=1.0, sr=44100)
        audio, sr = load_and_preprocess(wav_bytes)

        assert sr == 16000
        assert audio.ndim == 1
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6

    def test_denoise_audio(self):
        from src.pipeline import denoise_audio

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = denoise_audio(audio, 16000, prop_decrease=0.6)
        assert result.shape == audio.shape

    def test_cosine_fade(self):
        from src.pipeline import cosine_fade

        fade_in = cosine_fade(100, fade_in=True)
        assert len(fade_in) == 100
        assert fade_in[0] < fade_in[-1]

        fade_out = cosine_fade(100, fade_in=False)
        assert fade_out[0] > fade_out[-1]

    def test_find_overlaps_no_overlap(self):
        from src.pipeline import find_overlaps

        segments = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "B", "start": 1.0, "end": 2.0},
        ]
        non_overlaps, overlaps = find_overlaps(segments)

        assert len(overlaps) == 0
        assert len(non_overlaps) >= 2

    def test_find_overlaps_with_overlap(self):
        from src.pipeline import find_overlaps

        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 1.0, "end": 3.0},
        ]
        non_overlaps, overlaps = find_overlaps(segments)

        assert len(overlaps) > 0
        overlap = overlaps[0]
        assert overlap["start"] == pytest.approx(1.0)
        assert overlap["end"] == pytest.approx(2.0)

    def test_postprocess_track(self):
        from src.pipeline import postprocess_track

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        wav_bytes = postprocess_track(audio, 16000, peak_db=-1.0)

        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 16000
        assert np.max(np.abs(data)) <= 1.0

    def test_postprocess_segments_merge_same_speaker(self):
        from src.pipeline import postprocess_segments

        segments = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "A", "start": 1.1, "end": 2.0},  # gap=0.1s, same speaker
            {"speaker": "B", "start": 2.5, "end": 4.0},
        ]
        result = postprocess_segments(segments, max_gap_merge=0.3)
        assert len(result) == 2
        assert result[0]["speaker"] == "A"
        assert result[0]["end"] == 2.0

    def test_postprocess_segments_remove_islands(self):
        from src.pipeline import postprocess_segments

        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 2.0, "end": 2.1},  # 0.1s island
            {"speaker": "A", "start": 2.1, "end": 4.0},
        ]
        result = postprocess_segments(segments, min_island_duration=0.15)
        assert len(result) == 1
        assert result[0]["speaker"] == "A"

    def test_postprocess_segments_keep_real_turns(self):
        from src.pipeline import postprocess_segments

        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 2.0, "end": 5.0},
            {"speaker": "A", "start": 5.0, "end": 8.0},
        ]
        result = postprocess_segments(segments)
        assert len(result) == 3

    def test_assign_speakers_first_speaker(self):
        torch = pytest.importorskip("torch")
        from src.pipeline import assign_speakers

        segments = [
            {"speaker": "SPK_01", "start": 0.5, "end": 3.0},
            {"speaker": "SPK_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPK_01", "start": 3.5, "end": 5.0},
        ]
        embeddings = {
            "SPK_00": torch.randn(192),
            "SPK_01": torch.randn(192),
        }
        speaker_a, speaker_b = assign_speakers(segments, embeddings)
        assert speaker_a == "SPK_00"
        assert speaker_b == "SPK_01"


# ---------------------------------------------------------------------------
# Integration-level tests (require Modal — skip in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires Modal deployment and GPU")
class TestModalIntegration:
    def test_health_check(self):
        import modal

        Pipeline = modal.Cls.from_name("speaker-separation", "SpeakerSeparationPipeline")
        pipeline = Pipeline()
        status = pipeline.health_check.remote()
        assert "OK" in status

    def test_process_short_audio(self):
        import modal

        Pipeline = modal.Cls.from_name("speaker-separation", "SpeakerSeparationPipeline")
        pipeline = Pipeline()

        wav_bytes = make_wav_bytes(duration_s=5.0)
        results = pipeline.process_session.remote(wav_bytes, "test_session")

        assert "speaker_a" in results
        assert "speaker_b" in results
