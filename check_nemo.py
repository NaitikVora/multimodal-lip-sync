import modal
from pathlib import Path

app = modal.App("test-nemo")

@app.function(image=modal.Image.debian_slim(python_version="3.10").apt_install("ffmpeg", "libsndfile1").pip_install("nemo_toolkit[asr]==1.23.0", "pytorch-lightning", "transformers<4.38.0", "s3prl", "onnxruntime", "numpy<2", "huggingface_hub<0.22.0", "Cython==0.29.37", "soundfile"), gpu="A10G")
def test_nemo():
    import numpy as np
    import soundfile as sf
    import tempfile
    import json
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_path = tmpdir / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        manifest_path = tmpdir / "manifest.json"
        manifest_entry = {
            "audio_filepath": str(audio_path), "offset": 0.0, "duration": len(audio) / sr, "label": "infer", "text": "-", "num_speakers": 2, "rttm_filepath": None, "uem_filepath": None
        }
        manifest_path.write_text(json.dumps(manifest_entry) + "\n")

        cfg = OmegaConf.create({
            "num_workers": 0,
            "sample_rate": sr,
            "batch_size": 16,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(tmpdir),
                "oracle_vad": False,
                "oracle_num_speakers": False,
                "ignore_overlap": False,
                "collar": 0.25,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {"window_length_in_sec": 0.15, "shift_length_in_sec": 0.01, "smoothing": "median", "overlap": 0.875, "onset": 0.4, "offset": 0.35, "pad_onset": 0.05, "pad_offset": -0.1, "min_duration_on": 0.2, "min_duration_off": 0.2, "filter_speech_first": True},
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {"window_length_in_sec": [1.5], "shift_length_in_sec": [0.75], "multiscale_weights": [1], "save_embeddings": False},
                },
                "clustering": {
                    "parameters": {"oracle_num_speakers": True, "max_num_speakers": 2, "enhanced_count_thres": 80, "max_rp_threshold": 0.25, "sparse_search_volume": 30, "maj_vote_spk_count": False},
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephony",
                    "parameters": {"use_speaker_model_from_ckpt": True, "infer_batch_size": 25, "sigmoid_threshold": [0.7], "seq_eval_mode": False, "split_infer": True, "diar_eval_settings": [[0.25, False]]},
                },
            },
        })

        OmegaConf.set_struct(cfg, False)

        try:
            diarizer = ClusteringDiarizer(cfg=cfg)
            if not hasattr(diarizer, 'verbose'):
                diarizer.verbose = False
            diarizer.diarize()
            print("MSDD worked!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("MSDD failed:", e)

    return True
