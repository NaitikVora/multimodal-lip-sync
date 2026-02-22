# Speaker Separation Tool

A Python-based speaker separation tool for clinical audio recordings. Takes messy doctor–patient conversations (heavy cross-talk from closely-spaced microphones) and produces clean, speaker-separated output files.

All GPU computation runs on **Modal** (serverless GPU cloud). Your local Mac handles only CLI orchestration and file I/O.

---

## Quick Start

```bash
# 1. Install local dependencies
pip install -r requirements.txt

# 2. Authenticate with Modal (one-time)
modal setup

# 3. Create a HuggingFace secret for pyannote model access
#    First, accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
#    Then create a token at: https://huggingface.co/settings/tokens
modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxx

# 4. Deploy the Modal app (builds container image — first run takes ~5-10 min)
modal deploy modal_app.py

# 5. Process all sessions
python main.py --input-dir ./Audio_Files --output-dir ./output

# Process a single session
python main.py --input-dir ./Audio_Files --output-dir ./output --session 3

# Dry run (check files only)
python main.py --input-dir ./Audio_Files --output-dir ./output --dry-run

# Health check (verify models are loaded)
python main.py --input-dir ./Audio_Files --health-check
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **Modal account** | [modal.com](https://modal.com) — $30 free credit on signup |
| **HuggingFace account** | [huggingface.co](https://huggingface.co) — free |
| **HF Token** | Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **pyannote model access** | Accept terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  LOCAL MAC                                           │
│                                                      │
│  main.py (CLI entry point)                           │
│    ├── Reads WAV files from Audio_Files/             │
│    ├── Mixes multi-mic input to mono                 │
│    ├── Sends audio bytes to Modal GPU function       │
│    ├── Receives separated WAV bytes back             │
│    └── Saves output/ directory locally               │
└──────────────────────┬──────────────────────────────┘
                       │ modal.Cls.method.remote()
                       ▼
┌─────────────────────────────────────────────────────┐
│  MODAL CLOUD (A10G GPU, 24 GB VRAM)                  │
│                                                      │
│  SpeakerSeparationPipeline                           │
│    ├── @enter: Load all models once                  │
│    │   ├── pyannote diarization 3.1                  │
│    │   ├── ECAPA-TDNN speaker embeddings             │
│    │   └── SepFormer (whamr16k)                      │
│    │                                                  │
│    └── @method process_session()                     │
│        ├── Stage 0: Load, downmix, resample 16 kHz  │
│        ├── Stage 1: Denoise (noisereduce)            │
│        ├── Stage 2: Diarize (num_speakers=2)         │
│        ├── Stage 3: Extract speaker embeddings       │
│        ├── Stage 4: Hybrid separation                │
│        │   ├── Non-overlap → gate original audio     │
│        │   └── Overlap → SepFormer + emb. matching   │
│        ├── Stage 5: Stitch & normalize               │
│        └── Stage 6: Assign doctor vs patient         │
└─────────────────────────────────────────────────────┘
```

---

## Why Modal?

The ML models (pyannote, SepFormer, ECAPA-TDNN) require CUDA-capable GPUs. Since macOS doesn't have NVIDIA GPUs, we use **Modal** for on-demand serverless GPU access:

- **A10G GPU** with 24 GB VRAM — enough to hold all three models simultaneously
- **~$1.10/hr** — you only pay for actual GPU seconds
- **Container caching** — the image (with all models baked in) is cached after first build
- **Warm containers** — stays warm for 5 minutes between sessions, so subsequent calls start instantly
- **No DevOps** — no SSH, no Docker, no Kubernetes

---

## Pipeline Stages

### Stage 0 — Audio Loading & Preprocessing

Receives WAV bytes, reads with `soundfile`, downmixes multi-channel to mono, resamples to 16 kHz, and normalizes to [-1, 1].

**Why mono?** The two microphones are ~1 foot apart. At 16 kHz sample rate, the wavelength of the highest frequency is ~2 cm. At 1 foot (~30 cm) spacing, there's negligible spatial diversity above ~563 Hz (where the mic spacing equals half a wavelength). Beamforming or spatial separation methods won't help — the two channels are nearly identical copies with both speakers at similar levels on both.

### Stage 1 — Speech Enhancement / Denoising

`noisereduce` removes stationary noise (room tone, HVAC, computer fans). Conservative settings (`prop_decrease=0.6`) preserve speech quality. Skippable via config if it causes issues.

### Stage 2 — Speaker Diarization

`pyannote/speaker-diarization-3.1` identifies *who spoke when*. Constrained to `num_speakers=2` (doctor + patient). Also detects **overlap regions** where both speakers talk simultaneously.

### Stage 3 — Speaker Embedding Extraction

For each speaker, extracts ECAPA-TDNN embeddings from their 3-5 longest non-overlapping segments. Averages to get a robust centroid. These embeddings are used to solve the permutation problem in Stage 4.

### Stage 4 — Hybrid Separation

The key insight: **most conversation is non-overlapping** (~70-85% typically).

- **Non-overlapping segments**: Simply gate/copy the original audio into the correct speaker's track. Zero artifacts, perfect quality.
- **Overlapping segments**: Run SepFormer to separate the two sources, then match each source to a speaker using cosine similarity of their embeddings against the Stage 3 centroids.

15 ms cosine fades at segment boundaries prevent clicks.

### Stage 5 — Stitching & Post-processing

Both tracks are padded to the same length as the original. Silence when a speaker isn't talking. Normalized to -1 dB peak. Encoded as 16-bit PCM WAV at 16 kHz mono.

### Stage 6 — Doctor vs Patient Assignment

- **Primary heuristic**: Whoever speaks first is the doctor (clinicians typically initiate encounters)
- **Fallback**: If one speaker talks 3x more than the other, the shorter speaker is the doctor (patients tend to have longer symptom descriptions)

> **Note**: This is a heuristic. If the labels are swapped for your recordings, simply rename the output files.

---

## Libraries & Why

| Library | Role | Why this one |
|---|---|---|
| `pyannote.audio` | Speaker diarization | Best-in-class; native overlap detection; constrain to N speakers |
| `speechbrain` SepFormer | Source separation | State-of-the-art; `whamr16k` variant trained on noisy+reverberant audio |
| `speechbrain` ECAPA-TDNN | Speaker embeddings | Robust 192-dim embeddings; resolves SepFormer permutation ambiguity |
| `noisereduce` | Denoising | Pure Python; no complex dependencies; effective on stationary noise |
| `Modal` | Serverless GPU | On-demand A10G; no CUDA setup on Mac; ~$1.10/hr |
| `click` | CLI | Clean argument parsing with help text |
| `rich` | Terminal UI | Progress bars, tables, colored output |

---

## Key Design Decisions

### Why downmix to mono

Two mics 1 foot apart provide negligible spatial diversity. At 16 kHz, the max frequency is 8 kHz with a wavelength of ~4.3 cm. The mic spacing of ~30 cm is several wavelengths, causing spatial aliasing rather than useful spatial information. Mono simplifies all downstream models.

### Why hybrid gate + separate

Running a neural separator on the entire recording would introduce artifacts everywhere — including the 70-85% where only one person is talking. By gating non-overlapping segments directly from the (denoised) original, we preserve perfect audio quality where possible and only invoke SepFormer for the genuinely overlapping portions.

### Why sepformer-whamr16k over wsj02mix

The `wsj02mix` model was trained on clean, artificially mixed speech. Our clinical recordings have room reverberation, background noise, and compression artifacts. The `whamr16k` variant was trained on noisy + reverberant mixtures — a much better domain match.

### Why embedding-based permutation resolution

SepFormer's output order is arbitrary — "source 1" and "source 2" don't correspond to consistent speakers across chunks. By comparing each separated source's embedding to the speaker centroids from Stage 3, we assign sources to the correct speakers.

---

## Output Structure

```
output/
├── session_001_original.wav     # Original mixed recording
├── session_001_doctor.wav       # Doctor's voice isolated
├── session_001_patient.wav      # Patient's voice isolated
├── session_002_original.wav
├── session_002_doctor.wav
├── session_002_patient.wav
├── ...
└── session_010_patient.wav      # 30 files total
```

All outputs: 16 kHz, mono, 16-bit PCM WAV.

---

## File Transfer

Two strategies are implemented with automatic selection:

- **Direct byte passing** (default): Audio bytes are sent as function arguments and results returned directly. Simple and fast for files under 50 MB.
- **Volume-based transfer** (automatic fallback): For files exceeding 50 MB, audio is uploaded to a Modal Volume, processed, and results downloaded. This avoids Modal's function argument size limits.

---

## Challenges & Honest Assessment

- **Cross-talk from close mics is fundamentally hard.** Both speakers appear at similar levels on both channels. This limits what spatial/beamforming methods can achieve — we rely on learned separation instead.
- **Real-world audio ≠ synthetic benchmarks.** SepFormer achieves impressive SI-SNRi on WSJ0-2mix, but clinical recordings with room tone, compression artifacts, and variable mic placement are harder.
- **Overlap separation won't be studio-clean.** Expect artifacts in overlapping regions. Non-overlapping segments (the majority) will sound great since they're gated from the original.
- **Doctor/patient labeling is heuristic.** The first-speaker and speaking-time rules work for typical clinical encounters but aren't guaranteed.
- **Diarization can miss short utterances.** Brief backchannels ("mm-hmm", "okay") during the other speaker's turn may be misattributed.

---

## What I'd Improve With More Time

1. **Fine-tune models on clinical audio** — Domain adaptation would significantly improve diarization and separation quality
2. **pyannote PixIT pipeline** — Joint diarization + separation in a single model, potentially better overlap handling
3. **ClearerVoice-Studio MossFormer2** — Newer separation architecture that may outperform SepFormer on this domain
4. **Confidence scores per segment** — Flag low-confidence separations for human review
5. **Web UI** — Upload audio and preview results in the browser
6. **Streaming processing** — Process long recordings in chunks to reduce memory usage and enable progress tracking
7. **Speaker verification** — Use enrollment audio of the actual doctor for more reliable role assignment

---

## Cost Estimate

| Resource | Rate | Per Session | 10 Sessions |
|---|---|---|---|
| A10G GPU | $1.10/hr | ~$0.02-0.05 | ~$0.20-0.50 |
| Container startup | (first call only) | ~$0.02 | ~$0.02 |
| **Total** | | | **< $1.00** |

The container stays warm for 5 minutes between sessions, so sequential processing avoids repeated cold starts.

---

## Project Structure

```
├── README.md                # This file
├── requirements.txt         # Local dependencies (Mac)
├── config.py                # All configurable parameters
├── main.py                  # Local CLI entry point
├── modal_app.py             # Modal App, Image, GPU Cls definitions
├── src/
│   ├── __init__.py
│   ├── audio_utils.py       # Local audio I/O utilities
│   └── pipeline.py          # Full separation pipeline (runs in Modal container)
├── output/                  # Generated output files
├── Audio_Files/             # Input session directories
│   ├── Session_001/
│   ├── Session_002/
│   └── ...
└── tests/
    └── test_pipeline.py     # Unit + integration tests
```

---

## Troubleshooting

**"Modal app not deployed"** — Run `modal deploy modal_app.py` first.

**"HF_TOKEN not found"** — Create the secret: `modal secret create huggingface-token HF_TOKEN=hf_xxxxx`

**"Access to model denied"** — Accept the pyannote model terms at the HuggingFace links in Prerequisites.

**First deploy is slow (~5-10 min)** — Normal. Modal is building the container image and downloading ~3 GB of ML models. Subsequent deploys use the cached image.

**Timeout on long recordings** — The default timeout is 30 minutes. For recordings longer than ~45 minutes, increase `MODAL_TIMEOUT` in `config.py` and redeploy.

**Output labels are swapped** — The doctor/patient assignment is heuristic. Simply rename the files if they're wrong for your recording.
