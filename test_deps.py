import modal

_base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "libglib2.0-0", "git", "build-essential")
    .pip_install(
        "numpy<2", 
        "pyarrow<15.0.0",
        "huggingface_hub<0.22.0",
        "transformers<4.39.0",
        "torch==2.1.2", 
        "torchaudio==2.1.2",
        "pyannote.audio>=3.1.0,<3.4",
        "Cython==0.29.37", # Must be pre-installed for editdistance/youtokentome
        "omegaconf>=2.3.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.0",
    )
)

gpu_image = (
    _base_image
    # Install the full ASR toolkit to prevent missing editdistance/inflect/etc
    .pip_install("nemo_toolkit[asr]==1.23.0", "pytorch-lightning", "s3prl") 
    .run_commands("pip install git+https://github.com/wenet-e2e/wespeaker.git")
)

app = modal.App("test-deps")

@app.function(image=gpu_image)
def test_imports():
    from pyannote.audio import Pipeline
    import nemo.collections.asr.models
    import wespeaker
    import transformers
    return True
