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
        "huggingface_hub<0.24",
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

        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier

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
        """Run both Pipeline B and C in a single GPU call.

        Returns: {
            "channel": {"speaker_a": bytes, "speaker_b": bytes, "metadata": dict},
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
    def health_check(self) -> str:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        return (
            f"OK | device={self.device} | gpu={gpu_name} | "
            f"diarizer={'loaded' if self.diarizer else 'missing'} | "
            f"embedder={'loaded' if self.embedder else 'missing'} | "
            f"separator={'loaded' if self.separator else 'missing'}"
        )


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
