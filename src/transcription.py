"""Whisper-based transcription with speaker diarization labels.

Transcribes audio with word-level timestamps, then aligns each word
to the nearest diarization segment to produce speaker-labelled transcript.

Returns a list of utterance dicts:
  [{
    "speaker": "A",           # "A" or "B"
    "start": 12.34,           # seconds
    "end": 15.67,
    "text": "Hello, how are you doing today?"
  }, ...]
"""

import io
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Whisper Transcription
# ---------------------------------------------------------------------------


def transcribe_with_speakers(
    mixed_bytes: bytes,
    segments: list[dict],           # diarization segments
    speaker_a_label: str,
    speaker_b_label: str,
    sr: int = 16000,
    model_size: str = "large-v3",
    language: str | None = None,
    compute_type: str = "float16",
) -> list[dict]:
    """Transcribe mixed audio with word-level speaker attribution.

    Uses faster-whisper (CTranslate2) for efficient GPU inference,
    then aligns each word to the diarization timeline.

    Args:
        mixed_bytes: Raw WAV bytes of the mixed recording
        segments: Diarization segments [{"speaker", "start", "end"}]
        speaker_a_label: Internal diarization label for Speaker A
        speaker_b_label: Internal diarization label for Speaker B
        model_size: Whisper model size (default: large-v3)
        language: Force language (None = auto-detect)
        compute_type: "float16" for GPU, "float32" for CPU

    Returns:
        List of utterance dicts with speaker labels ("A" or "B")
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        # Fallback to openai-whisper
        return _transcribe_openai_whisper(
            mixed_bytes, segments, speaker_a_label, speaker_b_label, sr
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        compute_type = "float32"  # float16 not supported on CPU

    print(f"[Transcription] Loading Whisper {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Write to temp bytes buffer for faster-whisper
    audio_arr, _ = sf.read(io.BytesIO(mixed_bytes), dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1)

    print("[Transcription] Running transcription with word timestamps...")
    segs, info = model.transcribe(
        audio_arr,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    # Collect all words with timestamps
    words = []
    for seg in segs:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })

    print(f"[Transcription] Got {len(words)} words, aligning with {len(segments)} diarization segments...")

    # Assign speaker labels to each word
    label_map = {speaker_a_label: "A", speaker_b_label: "B"}
    word_speakers = []
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        speaker = _get_speaker_at_time(mid, segments, speaker_a_label)
        word_speakers.append({
            **word,
            "speaker_internal": speaker,
            "speaker": label_map.get(speaker, "A"),
        })

    # Group consecutive same-speaker words into utterances
    utterances = _group_words_into_utterances(word_speakers)
    print(f"[Transcription] Produced {len(utterances)} utterances")
    return utterances


def _transcribe_openai_whisper(
    mixed_bytes: bytes,
    segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    sr: int,
) -> list[dict]:
    """Fallback: openai-whisper with segment-level timestamps."""
    import whisper
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Transcription-fallback] Loading Whisper large-v3 on {device}...")
    model = whisper.load_model("large-v3", device=device)

    audio_arr, _ = sf.read(io.BytesIO(mixed_bytes), dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1)

    result = model.transcribe(audio_arr, word_timestamps=True, language=None)

    label_map = {speaker_a_label: "A", speaker_b_label: "B"}
    utterances = []

    for seg in result.get("segments", []):
        mid = (seg["start"] + seg["end"]) / 2
        speaker_internal = _get_speaker_at_time(mid, segments, speaker_a_label)
        utterances.append({
            "speaker": label_map.get(speaker_internal, "A"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    return utterances


def _get_speaker_at_time(
    t: float,
    segments: list[dict],
    default_speaker: str,
) -> str:
    """Return which speaker is active at time t (nearest segment midpoint)."""
    best_speaker = default_speaker
    best_overlap = -1.0

    for seg in segments:
        overlap = min(seg["end"], t + 0.001) - max(seg["start"], t - 0.001)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    # If no overlap found (gap), find the nearest segment
    if best_overlap <= 0:
        best_dist = float("inf")
        for seg in segments:
            dist = min(abs(seg["start"] - t), abs(seg["end"] - t))
            if dist < best_dist:
                best_dist = dist
                best_speaker = seg["speaker"]

    return best_speaker


def _group_words_into_utterances(
    words: list[dict],
    max_gap_s: float = 0.5,
    max_utterance_s: float = 20.0,
) -> list[dict]:
    """Group words into utterances by speaker and pause boundaries."""
    if not words:
        return []

    utterances = []
    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "words": [words[0]["word"]],
    }

    for word in words[1:]:
        gap = word["start"] - current["end"]
        duration = word["end"] - current["start"]

        # Start new utterance if: speaker changed, long pause, or max duration
        if (
            word["speaker"] != current["speaker"]
            or gap > max_gap_s
            or duration > max_utterance_s
        ):
            utterances.append({
                "speaker": current["speaker"],
                "start": current["start"],
                "end": current["end"],
                "text": " ".join(current["words"]).strip(),
            })
            current = {
                "speaker": word["speaker"],
                "start": word["start"],
                "end": word["end"],
                "words": [word["word"]],
            }
        else:
            current["end"] = word["end"]
            current["words"].append(word["word"])

    # Flush last utterance
    if current["words"]:
        utterances.append({
            "speaker": current["speaker"],
            "start": current["start"],
            "end": current["end"],
            "text": " ".join(current["words"]).strip(),
        })

    return utterances
