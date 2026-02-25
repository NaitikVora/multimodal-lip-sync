"""Modal app definition — defines the cloud-side Image, GPU Cls, and Volume.

Deploy with:  modal deploy modal_app.py

Pipelines:
  B: pyannote 3.1 + Channel-Aware + Wiener overlaps
  C: pyannote 3.1 + SepFormer overlaps
  D: NVIDIA NeMo MSDD + TitaNet + Channel-Aware (state-of-the-art meeting diarization)
  E: DiariZen (VoxSRC 2023 winner) + CAM++ embeddings + Channel-Aware

All pipelines include:
  - Denoising (noisereduce)
  - Diarization post-processing (merge, smooth, boundary refinement)
  - Channel-informed label correction
  - Cross-channel leakage estimation + spectral subtraction
  - VAD gating
  - Whisper large-v3 transcription with speaker labels
"""

import modal

# ---------------------------------------------------------------------------
# App & Infrastructure
# ---------------------------------------------------------------------------

app = modal.App("speaker-separation")

# Base GPU image with shared deps
_base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "libglib2.0-0", "git")
    .pip_install(
        "numpy<2",
        "pyarrow<15.0.0",
        "huggingface_hub<0.22.0",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "pyannote.audio>=3.1.0,<3.4",
        "speechbrain>=1.0.0,<1.1",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "noisereduce>=3.0.0",
        "scipy>=1.11.0,<2",
        "rich>=13.0.0",
        "scikit-learn>=1.3.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "Cython==0.29.37",
        "pytorch-lightning",
        # Transcription
        "faster-whisper>=1.0.0",
    )
)

# Single GPU image — NeMo and wespeaker are installed and imported lazily
gpu_image = (
    _base_image
    .pip_install("nemo_toolkit[asr]==1.23.0", "pytorch-lightning", "transformers<4.38.0", "s3prl", "onnxruntime", "openai-whisper", "peft==0.8.2")
    .run_commands("pip install git+https://github.com/wenet-e2e/wespeaker.git")
    .add_local_python_source("src")
)

audio_volume = modal.Volume.from_name("speaker-sep-audio", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")


# ---------------------------------------------------------------------------
# Primary GPU Pipeline Class (pyannote + SepFormer + Transcription)
# ---------------------------------------------------------------------------


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=2400,
    volumes={"/data": audio_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
class SpeakerSeparationPipeline:
    """Loads ML models once on container start, then processes sessions.

    Models loaded:
      - pyannote/speaker-diarization-3.1
      - speechbrain/spkrec-ecapa-voxceleb  (ECAPA-TDNN embeddings)
      - speechbrain/sepformer-whamr16k     (SepFormer source separation)
      - faster-whisper large-v3            (transcription)
    """

    @modal.enter()
    def load_models(self):
        import os
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier
        from faster_whisper import WhisperModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading pyannote diarization pipeline...")
        self.diarizer = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ["HF_TOKEN"],
        )
        self.diarizer.to(self.device)

        print("Loading SpeechBrain ECAPA-TDNN speaker embeddings...")
        self.embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

        print("Loading SpeechBrain SepFormer (whamr16k)...")
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            run_opts={"device": str(self.device)},
        )

        print("Loading Whisper large-v3...")
        compute_type = "float16" if str(self.device) == "cuda" else "float32"
        self.whisper = WhisperModel(
            "large-v3",
            device=str(self.device),
            compute_type=compute_type,
        )

        print("All models loaded successfully!")

    @modal.method()
    def process_session(
        self,
        channel_a_bytes: bytes,
        channel_b_bytes: bytes,
        mixed_bytes: bytes,
        session_name: str,
    ) -> dict:
        """Run Pipeline B (channel-aware) only.

        Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
        """
        from src.pipeline import run_pipeline

        return run_pipeline(
            channel_a_bytes=channel_a_bytes,
            channel_b_bytes=channel_b_bytes,
            mixed_bytes=mixed_bytes,
            session_name=session_name,
            diarizer=self.diarizer,
            embedder=self.embedder,
            separator=self.separator,
        )

    @modal.method()
    def process_session_compare(
        self,
        channel_a_bytes: bytes,
        channel_b_bytes: bytes,
        mixed_bytes: bytes,
        session_name: str,
    ) -> dict:
        """Run Pipelines B and C in a single GPU call (pyannote-based).

        Returns: {
            "channel":   {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict},
            "sepformer": {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict},
        }
        """
        from src.pipeline import run_pipeline, run_pipeline_sepformer

        channel_results = run_pipeline(
            channel_a_bytes=channel_a_bytes,
            channel_b_bytes=channel_b_bytes,
            mixed_bytes=mixed_bytes,
            session_name=session_name,
            diarizer=self.diarizer,
            embedder=self.embedder,
            separator=self.separator,
        )

        sepformer_results = run_pipeline_sepformer(
            channel_a_bytes=channel_a_bytes,
            channel_b_bytes=channel_b_bytes,
            mixed_bytes=mixed_bytes,
            session_name=session_name,
            diarizer=self.diarizer,
            embedder=self.embedder,
            separator=self.separator,
        )

        return {
            "channel": channel_results,
            "sepformer": sepformer_results,
        }

    @modal.method()
    def process_all_pipelines(
        self,
        channel_a_bytes: bytes,
        channel_b_bytes: bytes,
        mixed_bytes: bytes,
        session_name: str,
    ) -> dict:
        """Run ALL 5 pipelines: energy (local), B, C, D (NeMo), E (DiariZen).

        Returns dict keyed by pipeline ID.
        """
        from src.pipeline import run_pipeline, run_pipeline_sepformer
        from src.nemo_pipeline import run_pipeline_nemo
        from src.diarizen_pipeline import run_pipeline_diarizen

        results = {}

        # Pipeline B: pyannote + channel-aware
        try:
            results["channel"] = run_pipeline(
                channel_a_bytes=channel_a_bytes,
                channel_b_bytes=channel_b_bytes,
                mixed_bytes=mixed_bytes,
                session_name=session_name,
                diarizer=self.diarizer,
                embedder=self.embedder,
                separator=self.separator,
            )
        except Exception as e:
            print(f"[Pipeline B] Failed: {e}")
            results["channel"] = {"speaker_a": b"", "speaker_b": b"", "metadata": {"error": str(e)}}

        # Pipeline C: pyannote + SepFormer
        try:
            results["sepformer"] = run_pipeline_sepformer(
                channel_a_bytes=channel_a_bytes,
                channel_b_bytes=channel_b_bytes,
                mixed_bytes=mixed_bytes,
                session_name=session_name,
                diarizer=self.diarizer,
                embedder=self.embedder,
                separator=self.separator,
            )
        except Exception as e:
            print(f"[Pipeline C] Failed: {e}")
            results["sepformer"] = {"speaker_a": b"", "speaker_b": b"", "metadata": {"error": str(e)}}

        # Pipeline D: NeMo MSDD + TitaNet
        try:
            results["nemo"] = run_pipeline_nemo(
                channel_a_bytes=channel_a_bytes,
                channel_b_bytes=channel_b_bytes,
                mixed_bytes=mixed_bytes,
                session_name=session_name,
            )
        except Exception as e:
            print(f"[Pipeline D - NeMo] Failed: {e}")
            results["nemo"] = {"speaker_a": b"", "speaker_b": b"", "metadata": {"error": str(e)}}

        # Pipeline E: DiariZen
        try:
            results["diarizen"] = run_pipeline_diarizen(
                channel_a_bytes=channel_a_bytes,
                channel_b_bytes=channel_b_bytes,
                mixed_bytes=mixed_bytes,
                session_name=session_name,
            )
        except Exception as e:
            print(f"[Pipeline E - DiariZen] Failed: {e}")
            results["diarizen"] = {"speaker_a": b"", "speaker_b": b"", "metadata": {"error": str(e)}}

        return results

    @modal.method()
    def process_diarizen_only(
        self,
        channel_a_bytes: bytes,
        channel_b_bytes: bytes,
        mixed_bytes: bytes,
        session_name: str,
    ) -> dict:
        """Run only Pipeline E (DiariZen) for a single session.

        Returns: {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict}
        """
        from src.diarizen_pipeline import run_pipeline_diarizen

        try:
            return run_pipeline_diarizen(
                channel_a_bytes=channel_a_bytes,
                channel_b_bytes=channel_b_bytes,
                mixed_bytes=mixed_bytes,
                session_name=session_name,
            )
        except Exception as e:
            print(f"[Pipeline E - DiariZen] Failed: {e}")
            return {"speaker_a": b"", "speaker_b": b"", "metadata": {"error": str(e)}}

    @modal.method()
    def transcribe_session(
        self,
        channel_a_bytes: bytes,
        channel_b_bytes: bytes,
        mixed_bytes: bytes,
        session_name: str,
        language: str | None = None,
    ) -> dict:
        """Transcribe with Whisper large-v3, aligned to pyannote diarization.

        Returns:
        {
            "utterances": [{"speaker": "A"|"B", "start": float, "end": float, "text": str}],
            "language": str,
            "duration_s": float,
        }
        """
        from src.pipeline import (
            load_and_preprocess,
            denoise_audio,
            diarize,
            postprocess_segments,
            extract_speaker_embeddings,
            assign_speakers,
            find_overlaps,
        )
        from src.transcription import transcribe_with_speakers
        import numpy as np

        print(f"\n[Transcription] {session_name}")

        # Load + preprocess
        ch_a_audio, sr = load_and_preprocess(channel_a_bytes)
        ch_b_audio, _ = load_and_preprocess(channel_b_bytes)
        mixed_audio, _ = load_and_preprocess(mixed_bytes)

        mixed_clean = denoise_audio(mixed_audio, sr, 0.6)
        ch_a_clean = denoise_audio(ch_a_audio, sr, 0.6)
        ch_b_clean = denoise_audio(ch_b_audio, sr, 0.6)

        # Diarize with pyannote
        raw_segments = diarize(mixed_clean, sr, self.diarizer, num_speakers=2)
        segments = postprocess_segments(raw_segments, mixed_clean, sr)

        # Assign speaker labels
        non_overlaps_raw, _ = find_overlaps(segments)
        speaker_embeddings = extract_speaker_embeddings(mixed_clean, sr, non_overlaps_raw, self.embedder)
        speaker_a_label, speaker_b_label = assign_speakers(
            segments, speaker_embeddings, ch_a_clean, ch_b_clean, sr,
        )

        # Transcribe with pre-loaded Whisper
        utterances = _transcribe_with_preloaded_whisper(
            self.whisper, mixed_bytes, segments, speaker_a_label, speaker_b_label, language
        )

        return {
            "utterances": utterances,
            "language": language or "auto",
            "duration_s": round(len(mixed_audio) / sr, 1),
            "num_segments": len(segments),
        }

    @modal.method()
    def health_check(self) -> str:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        return (
            f"OK | device={self.device} | gpu={gpu_name} | "
            f"diarizer={'loaded' if self.diarizer else 'missing'} | "
            f"embedder={'loaded' if self.embedder else 'missing'} | "
            f"separator={'loaded' if self.separator else 'missing'} | "
            f"whisper={'loaded' if self.whisper else 'missing'}"
        )


def _transcribe_with_preloaded_whisper(
    whisper_model,
    mixed_bytes: bytes,
    segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    language: str | None,
) -> list[dict]:
    """Helper: transcribe using already-loaded WhisperModel."""
    import io
    import numpy as np
    import soundfile as sf

    audio_arr, sr = sf.read(io.BytesIO(mixed_bytes), dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1)

    segs_iter, info = whisper_model.transcribe(
        audio_arr,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    words = []
    for seg in segs_iter:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    from src.transcription import _get_speaker_at_time, _group_words_into_utterances

    label_map = {speaker_a_label: "A", speaker_b_label: "B"}
    word_speakers = []
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        spk_internal = _get_speaker_at_time(mid, segments, speaker_a_label)
        word_speakers.append({
            **word,
            "speaker": label_map.get(spk_internal, "A"),
        })

    return _group_words_into_utterances(word_speakers)


# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------


@app.function(image=gpu_image, volumes={"/data": audio_volume}, timeout=60)
def upload_to_volume(data: bytes, remote_path: str) -> str:
    from pathlib import Path

    path = Path(f"/data/{remote_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    audio_volume.commit()
    return str(path)


@app.function(image=gpu_image, volumes={"/data": audio_volume}, timeout=60)
def download_from_volume(remote_path: str) -> bytes:
    from pathlib import Path

    path = Path(f"/data/{remote_path}")
    return path.read_bytes()
