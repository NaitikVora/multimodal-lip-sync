"""Modal app definition — defines the cloud-side Image, GPU Cls, and Volume.

Deploy with:  modal deploy modal_app.py
"""

import modal

# ---------------------------------------------------------------------------
# App & Infrastructure
# ---------------------------------------------------------------------------

app = modal.App("speaker-separation")

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "numpy<2",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "pyannote.audio>=3.1.0,<3.4",
        "speechbrain>=1.0.0,<1.1",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "noisereduce>=3.0.0",
        "scipy>=1.11.0,<2",
        "rich>=13.0.0",
    )
    .add_local_python_source("src")
)

audio_volume = modal.Volume.from_name("speaker-sep-audio", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")

# ---------------------------------------------------------------------------
# GPU Pipeline Class
# ---------------------------------------------------------------------------


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": audio_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
class SpeakerSeparationPipeline:
    """Loads ML models once on container start, then processes sessions."""

    @modal.enter()
    def load_models(self):
        import os

        import huggingface_hub
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier

        # Login so all HF downloads are authenticated (pyannote needs gated model access)
        huggingface_hub.login(token=os.environ["HF_TOKEN"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading pyannote diarization pipeline...")
        self.diarizer = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
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

        print("All models loaded successfully!")

    @modal.method()
    def process_session(self, audio_bytes: bytes, session_name: str) -> dict:
        """Process a single session's audio.

        Args:
            audio_bytes: Raw WAV file bytes (mixed/mono/stereo).
            session_name: Human-readable session identifier.

        Returns:
            {"original": bytes, "doctor": bytes, "patient": bytes}
        """
        from src.pipeline import run_pipeline

        return run_pipeline(
            audio_bytes=audio_bytes,
            session_name=session_name,
            diarizer=self.diarizer,
            embedder=self.embedder,
            separator=self.separator,
            denoise_enabled=True,
            denoise_prop=0.6,
            num_speakers=2,
            num_enrollment=5,
            min_enrollment_duration=1.0,
            fade_ms=15,
            peak_db=-1.0,
        )

    @modal.method()
    def health_check(self) -> str:
        """Quick check that models are loaded and GPU is available."""
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        return (
            f"OK | device={self.device} | gpu={gpu_name} | "
            f"diarizer={'loaded' if self.diarizer else 'missing'} | "
            f"embedder={'loaded' if self.embedder else 'missing'} | "
            f"separator={'loaded' if self.separator else 'missing'}"
        )


# ---------------------------------------------------------------------------
# Volume-based transfer for large files (Option B fallback)
# ---------------------------------------------------------------------------


@app.function(image=gpu_image, volumes={"/data": audio_volume}, timeout=60)
def upload_to_volume(data: bytes, remote_path: str) -> str:
    """Write bytes to the shared Modal Volume."""
    from pathlib import Path

    path = Path(f"/data/{remote_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    audio_volume.commit()
    return str(path)


@app.function(image=gpu_image, volumes={"/data": audio_volume}, timeout=60)
def download_from_volume(remote_path: str) -> bytes:
    """Read bytes from the shared Modal Volume."""
    from pathlib import Path

    path = Path(f"/data/{remote_path}")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Volume-based processing for large files
# ---------------------------------------------------------------------------


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/data": audio_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
class SpeakerSeparationPipelineVolume:
    """Same as SpeakerSeparationPipeline but reads/writes via Volume."""

    @modal.enter()
    def load_models(self):
        import os

        import huggingface_hub
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier

        huggingface_hub.login(token=os.environ["HF_TOKEN"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diarizer = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        self.diarizer.to(self.device)

        self.embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            run_opts={"device": str(self.device)},
        )

    @modal.method()
    def process_session_volume(
        self, input_path: str, output_prefix: str, session_name: str
    ) -> dict:
        """Process from Volume path, write results back, return output paths."""
        from pathlib import Path

        from src.pipeline import run_pipeline

        audio_bytes = Path(f"/data/{input_path}").read_bytes()

        results = run_pipeline(
            audio_bytes=audio_bytes,
            session_name=session_name,
            diarizer=self.diarizer,
            embedder=self.embedder,
            separator=self.separator,
        )

        output_paths = {}
        for key, wav_bytes in results.items():
            out_path = f"{output_prefix}_{key}.wav"
            Path(f"/data/{out_path}").parent.mkdir(parents=True, exist_ok=True)
            Path(f"/data/{out_path}").write_bytes(wav_bytes)
            output_paths[key] = out_path

        audio_volume.commit()
        return output_paths
