<h1 align="center">SpeakerLens</h1>

<p align="center">
  <b>Multi-pipeline speaker separation & diarization for clinical audio</b><br>
  <sub>5 pipelines &middot; 11-stage signal processing &middot; GPU-accelerated on Modal &middot; React web UI</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Modal-A10G_GPU-7C3AED?style=flat" alt="Modal">
  <img src="https://img.shields.io/badge/React-19.2-61DAFB?style=flat&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
</p>

---

Takes messy doctor-patient conversations recorded on closely-spaced microphones (heavy cross-talk, overlapping speech, room noise) and produces clean, speaker-isolated audio tracks. Runs **5 competing separation pipelines** in parallel on a single A10G GPU, each using different diarization and separation strategies, so you can compare results and pick the best output for your use case.

## Table of Contents

- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [The Five Pipelines](#the-five-pipelines)
- [Pipeline Comparison](#pipeline-comparison)
- [11-Stage Signal Processing Deep Dive](#11-stage-signal-processing-deep-dive)
- [Diarization Models](#diarization-models)
- [Separation Algorithms](#separation-algorithms)
- [Transcription Engine](#transcription-engine)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [Frontend Architecture](#frontend-architecture)
- [Modal GPU Deployment](#modal-gpu-deployment)
- [Configuration Reference](#configuration-reference)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Challenges & Honest Assessment](#challenges--honest-assessment)
- [Cost Estimate](#cost-estimate)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BROWSER  (React 19 + TypeScript + Tailwind CSS 4 + WaveSurfer.js)    │
│                                                                         │
│  ┌────────────┐  ┌────────────────┐  ┌─────────────────────────────┐  │
│  │ Upload     │  │ Session Cards  │  │ Waveform Players (A/B)      │  │
│  │ Zone       │  │ + Pipeline     │  │ + Transcript Panel          │  │
│  │ (drag/drop)│  │   Tabs         │  │ + Model Comparison          │  │
│  └────────────┘  └────────────────┘  └─────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ REST + Server-Sent Events (SSE)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL SERVER  (FastAPI — server.py, port 8000)                         │
│                                                                         │
│  POST /api/upload           — receive WAV files                         │
│  GET  /api/sessions         — discover Audio_Files/                      │
│  POST /api/process/:id      — trigger all 5 pipelines on Modal          │
│  GET  /api/process/:id/status — SSE stream (real-time progress)         │
│  GET  /api/results/:id      — serve pipeline outputs + metadata         │
│  POST /api/transcribe/:id   — trigger Whisper transcription             │
│  GET  /api/transcript/:id   — fetch diarized transcript JSON            │
│  GET  /api/health           — Modal + model availability check          │
│  GET  /api/pipelines        — list available pipeline configurations    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ modal.Cls.method.remote()
                             │ (byte transfer < 50 MB, Volume > 50 MB)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODAL CLOUD  (NVIDIA A10G — 24 GB VRAM, 2400s timeout)                │
│                                                                         │
│  SpeakerSeparationPipeline (@modal.cls)                                 │
│    │                                                                    │
│    ├── @enter (cold start ~15s, models pre-loaded in image)             │
│    │   ├── pyannote/speaker-diarization-3.1      (diarization)          │
│    │   ├── speechbrain/spkrec-ecapa-voxceleb     (192-dim embeddings)   │
│    │   ├── speechbrain/sepformer-whamr16k        (neural separation)    │
│    │   └── faster-whisper large-v3               (transcription, fp16)  │
│    │                                                                    │
│    ├── process_all_pipelines()   → run A/B/C/D/E in parallel            │
│    ├── process_session()         → Pipeline B only                      │
│    ├── process_session_compare() → Pipelines B + C                      │
│    ├── process_diarizen_only()   → Pipeline E only                      │
│    ├── transcribe_session()      → Whisper + diarization alignment      │
│    └── health_check()            → GPU + model verification             │
│                                                                         │
│  Pipeline A: Energy-Based         (src/energy_pipeline.py)  — CPU only  │
│  Pipeline B: Pyannote + Wiener    (src/pipeline.py)         — GPU       │
│  Pipeline C: Pyannote + SepFormer (src/pipeline.py)         — GPU       │
│  Pipeline D: NeMo MSDD + TitaNet  (src/nemo_pipeline.py)   — GPU       │
│  Pipeline E: DiariZen + CAM++     (src/diarizen_pipeline.py) — GPU      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **Node.js** | 18+ (for frontend) |
| **Modal account** | [modal.com](https://modal.com) — $30 free credit on signup |
| **HuggingFace token** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **pyannote access** | Accept terms: [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |

### Setup

```bash
# Python dependencies
pip install -r requirements.txt

# Modal setup (one-time)
modal setup
modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxx

# Deploy GPU pipeline (first run ~5-10 min to build image + download ~3 GB models)
modal deploy modal_app.py

# Frontend
cd frontend && npm install && npm run build && cd ..

# Launch
python server.py
# Open http://localhost:8000
```

### Development Mode

```bash
# Terminal 1 — backend
python server.py

# Terminal 2 — frontend with HMR
cd frontend && npm run dev
# Open http://localhost:5173 (Vite proxies /api to FastAPI)
```

### CLI (alternative)

```bash
python main.py --input-dir ./Audio_Files --output-dir ./output
python main.py --input-dir ./Audio_Files --output-dir ./output --session 3
python main.py --input-dir ./Audio_Files --output-dir ./output --dry-run
```

---

## The Five Pipelines

Each pipeline uses a different combination of diarization model and separation strategy. All share the same 11-stage signal processing chain (preprocessing, denoising, post-processing, leakage suppression, VAD gating, normalization) but differ in how they determine *who spoke when* and how they handle overlapping speech.

### Pipeline A — Energy-Based (Baseline)

| Property | Value |
|---|---|
| **File** | `src/energy_pipeline.py` |
| **GPU required** | No (CPU only) |
| **Diarization** | None (energy ratio per frame) |
| **Separation** | Soft spectral masking |
| **Speed** | < 1 second |
| **Best for** | Quick baseline, real-time preview |

**Algorithm:** Frame-level RMS energy comparison between channels. 25ms frames, 10ms hop. Computes soft mask `a_ratio = energy_a / (energy_a + energy_b)`, sharpens with `clamp(ratio * 1.5 - 0.25, 0, 1)`, applies per-sample. 15ms cosine fades at silence-to-speech transitions.

### Pipeline B — Pyannote 3.1 + Channel-Aware Wiener

| Property | Value |
|---|---|
| **File** | `src/pipeline.py` → `run_pipeline()` |
| **Diarization model** | `pyannote/speaker-diarization-3.1` |
| **Embedding model** | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-dim) |
| **Overlap handling** | Wiener filtering with learned speaker spectral profiles |
| **Leakage suppression** | Spectral subtraction (alpha=0.7, floor=0.08) |
| **Speed** | 30-120s per session |

**Key innovation:** Non-overlapping segments (70-85% of recording) are extracted directly from the primary channel with spectral leakage suppression — zero neural artifacts. Only overlap regions use Wiener spectral masking with pre-computed speaker frequency profiles.

### Pipeline C — Pyannote 3.1 + SepFormer Neural Separation

| Property | Value |
|---|---|
| **File** | `src/pipeline.py` → `run_pipeline_sepformer()` |
| **Diarization model** | `pyannote/speaker-diarization-3.1` |
| **Separation model** | `speechbrain/sepformer-whamr16k` (Transformer, 2-source) |
| **Embedding model** | ECAPA-TDNN (permutation resolution) |
| **Overlap handling** | SepFormer + Wiener refinement |
| **Speed** | 60-180s per session |

**Key difference from B:** Overlapping segments go through SepFormer (a dual-path Transformer trained on WHAM-R noisy+reverberant mixtures) instead of basic Wiener masking. Embedding-based permutation resolution matches SepFormer's arbitrary output order to the correct speakers. Wiener refinement applies as a final cleanup pass.

### Pipeline D — NeMo MSDD + TitaNet

| Property | Value |
|---|---|
| **File** | `src/nemo_pipeline.py` |
| **Diarization** | NeMo ClusteringDiarizer with Multi-Scale Diarization Decoder |
| **VAD model** | `vad_multilingual_marblenet` |
| **Embedding model** | `titanet_large` (multi-scale: 1.5s, 1.25s, 1.0s, 0.75s, 0.5s windows) |
| **Clustering** | Agglomerative + MSDD refinement |
| **Separation** | Channel-aware Wiener (same as Pipeline B) |
| **Speed** | 120-300s per session |

**MSDD configuration:**
```
model_path:       diar_msdd_telephony
sigmoid_threshold: [0.7]
batch_size:        25
split_infer:       True
```

**Multi-scale embedding windows** — TitaNet extracts embeddings at 5 different temporal scales (0.5s to 1.5s) with corresponding shifts (0.1s to 0.75s), then MSDD fuses them for more robust speaker decisions. The `enhanced_count_threshold=80` and `max_rp_threshold=0.25` control cluster merging behavior.

### Pipeline E — DiariZen (VoxSRC 2023 Winner)

| Property | Value |
|---|---|
| **File** | `src/diarizen_pipeline.py` |
| **Diarization** | DiariZen end-to-end system (BUT-FIT, VoxSRC 2023 winner) |
| **Fallback embeddings** | wespeaker CAM++ (English) |
| **Clustering** | Spectral clustering with AgglomerativeClustering (cosine, average linkage) |
| **Separation** | Channel-aware Wiener (same as Pipeline B) |
| **Speed** | 120-240s per session |

**Fallback path:** If DiariZen module is unavailable, uses wespeaker's CAM++ model with sliding-window segmentation (1.5s window, 0.5s hop), extracts embeddings per window, and performs agglomerative clustering. Consecutive same-speaker segments with < 0.5s gap are merged.

---

## Pipeline Comparison

| | Pipeline A | Pipeline B | Pipeline C | Pipeline D | Pipeline E |
|---|---|---|---|---|---|
| **Approach** | Signal processing | Diarize + mask | Diarize + neural sep | MSDD diarize + mask | E2E diarize + mask |
| **Diarizer** | None | pyannote 3.1 | pyannote 3.1 | NeMo MSDD | DiariZen |
| **Overlap method** | Energy ratio | Wiener mask | SepFormer | Wiener mask | Wiener mask |
| **GPU required** | No | Yes | Yes | Yes | Yes |
| **Latency** | <1s | 30-120s | 60-180s | 120-300s | 120-240s |
| **Non-overlap quality** | Good | Excellent | Excellent | Excellent | Excellent |
| **Overlap quality** | Poor | Good | Best | Good | Good |
| **Diarization accuracy** | N/A | High | High | Highest (multi-scale) | High (E2E) |
| **Model size** | 0 | ~800 MB | ~1.2 GB | ~2 GB | ~600 MB |
| **Best for** | Quick preview | General use | Heavy overlap | Telephony/meetings | Research comparison |

**Key tradeoff:** Pipelines B-E all produce identical quality on non-overlapping speech (70-85% of a typical conversation) because they use the same channel-aware leakage suppression. The differences only appear in overlapping regions — where Pipeline C (SepFormer) has the strongest separation, while B/D/E rely on spectral masking.

---

## 11-Stage Signal Processing Deep Dive

All GPU pipelines (B through E) share this processing chain. The diarization model differs, but everything else is identical.

### Stage 0 — Audio Loading & Preprocessing

```
Input WAV bytes → soundfile.read() → downmix to mono → resample to 16 kHz → normalize to [-1, 1]
```

- **Downmix:** `np.mean(audio, axis=1)` for multi-channel
- **Resample:** librosa with polyphase filtering to 16 kHz
- **Normalize:** `audio / np.max(np.abs(audio))`

**Why mono?** Two microphones ~30 cm apart. At 16 kHz (max freq 8 kHz, wavelength 4.3 cm), the mic spacing is ~7 wavelengths — well into spatial aliasing territory. There's no usable spatial diversity above ~563 Hz. Beamforming is futile; both channels are near-identical copies with both speakers at similar levels.

### Stage 1 — Denoising

| Parameter | Value | Rationale |
|---|---|---|
| `prop_decrease` | 0.6 | Conservative — preserve speech, 60% noise reduction |
| `n_fft` | 2048 | Better frequency resolution than default 1024 |
| `freq_mask_smooth_hz` | 500 | Smooth spectral mask, reduce musical noise |
| `time_mask_smooth_ms` | 50 | Smooth temporal mask, reduce flutter |

Targets stationary noise (HVAC, room tone, computer fans). Conservative to avoid speech distortion. Skippable via `DENOISE_ENABLED=False`.

### Stage 2 — Speaker Diarization

Model-specific (see [Diarization Models](#diarization-models)). All constrained to `num_speakers=2`. Outputs list of `{speaker, start, end}` segments with overlap detection.

### Stage 3 — Diarization Post-Processing

Five-pass segment refinement:

| Pass | Operation | Threshold |
|---|---|---|
| 1 | Merge same-speaker gaps | < 0.3s gap |
| 2 | Remove isolated islands | < 0.15s duration |
| 3 | Absorb short segments into neighbors | < 0.25s duration |
| 4 | Final same-speaker merge | Adjacent same-speaker |
| 5 | Boundary snap to energy minima | ±30ms search window |

**Boundary snapping algorithm:** For each segment boundary, analyze a ±30ms window. Compute short-time energy in 3ms frames. Snap to the frame with minimum energy (natural speech pause). This prevents mid-syllable cuts and improves perceived separation quality.

### Stage 4 — Channel-Informed Diarization Correction

For each diarized segment, compute the energy ratio between channels during solo speech:

```
channel_a_ratio = energy(channel_a[segment]) / (energy(channel_a[segment]) + energy(channel_b[segment]))
```

If Speaker A consistently has `ratio < 0.4` and Speaker B has `ratio > 0.6`, the diarization labels are swapped relative to the physical microphone positions — correct them. Fixes mic misidentification.

### Stage 5 — Speaker Embedding Extraction (ECAPA-TDNN)

| Parameter | Value |
|---|---|
| Enrollment segments per speaker | Up to 8 (longest non-overlapping) |
| Minimum segment duration | 0.5s |
| Embedding dimensionality | 192 |
| Aggregation | Duration-weighted average |

```
centroid = Σ(embedding_i × duration_i) / Σ(duration_i)
```

Longer segments contribute more to the centroid — they're more reliable representations.

### Stage 6 — Speaker Label Assignment

| Strategy | Rule | Rationale |
|---|---|---|
| Primary | First speaker = Speaker A | Clinicians typically initiate encounters |
| Fallback | If talk-time ratio > 3:1, shorter speaker = A | Patients have longer symptom descriptions |

### Stage 7 — Cross-Channel Leakage Estimation

During non-overlapping segments only:

```
leak_a2b = median(energy_b / energy_a)    for Speaker A solo segments
leak_b2a = median(energy_a / energy_b)    for Speaker B solo segments
```

Clipped to `[0.0, 0.95]`. Default fallback: `0.5` (50% leakage — typical for mics ~1 foot apart with speakers at similar distances).

### Stage 8 — Speaker Spectral Profile Computation

| Parameter | Value |
|---|---|
| FFT size | 1024 (64ms window @ 16 kHz) |
| Hop length | 256 (16ms frame shift) |
| Profile segments | Top 15 longest non-overlapping per speaker |
| Profile dimension | 513 frequency bins (n_fft/2 + 1) |

For each speaker: compute STFT magnitude `|S|` across their longest solo segments, average to get a 513-dim frequency fingerprint. Used for Wiener masking in overlap regions.

### Stage 9 — Separation (Hybrid Gate + Separate)

**Non-overlapping segments (70-85% of recording):**

Extract from primary channel, apply spectral leakage suppression:

```
|S_clean| = max(|S_primary| - α × leak_ratio × |S_secondary|,  floor × |S_primary|)
S_clean = |S_clean| × exp(i × phase_primary)
```

| Parameter | Value | Rationale |
|---|---|---|
| `alpha` | 0.7 | 70% of estimated leakage subtracted; 30% safety margin |
| `floor` | 0.08 | Keep 8% minimum — prevents musical noise artifacts |
| Phase | Preserved from primary | Phase modification causes perceptible distortion |

**Overlapping segments — Wiener masking (Pipelines B, D, E):**

```
mask_a = |profile_a|² / (|profile_a|² + |profile_b|² + ε)
mask_b = |profile_b|² / (|profile_a|² + |profile_b|² + ε)
S_a = S_overlap × mask_a
S_b = S_overlap × mask_b
```

**Overlapping segments — SepFormer (Pipeline C):**

| Parameter | Value |
|---|---|
| Model | `speechbrain/sepformer-whamr16k` |
| Minimum chunk | 0.5s (zero-pad if shorter) |
| Context padding | ±0.25s surrounding audio |
| Output assignment | Cosine similarity to speaker centroids |

```
score_normal  = cos(src1, centroid_A) + cos(src2, centroid_B)
score_swapped = cos(src1, centroid_B) + cos(src2, centroid_A)
if score_swapped > score_normal: swap src1, src2
```

**Segment transitions — Cosine fading:**

20ms cosine fade-in/fade-out at every segment boundary. Overlap-add stitching prevents audible clicks:

```
fade_in  = sin(t × π/2)    for t ∈ [0, 1] over 20ms
fade_out = cos(t × π/2)    for t ∈ [0, 1] over 20ms
```

### Stage 10 — VAD Gating (Post-Separation Cleanup)

Energy-based voice activity detection applied after separation to silence residual bleed:

| Parameter | Value | Rationale |
|---|---|---|
| Frame length | 10ms | Fine temporal resolution |
| Threshold | -40 dB below track peak | Only gate very quiet residuals |
| Attack | 5ms | Fast fade-in when speech starts |
| Release | 80ms | Slow fade-out preserves natural speech tails |

```
gate = interpolate(frame_envelope, sample_rate)
output = separated_audio × gate
```

### Stage 11 — Normalization & Encoding

| Property | Value |
|---|---|
| Peak normalization | -1.0 dB (99.5% of full scale) |
| Sample rate | 16 kHz |
| Bit depth | 16-bit PCM |
| Channels | Mono |
| Format | WAV |

---

## Diarization Models

### pyannote/speaker-diarization-3.1 (Pipelines B, C)

- **Architecture:** Segmentation model (PyanNet) + embedding extraction + agglomerative clustering
- **Segmentation:** Trained on VoxConverse, AMI, DIHARD — 5s sliding window, 0.5s step
- **Overlap detection:** Native multi-label output (each frame can have 0, 1, or 2+ active speakers)
- **Embedding:** X-vector based, 512-dim
- **Clustering:** Agglomerative with centroid linkage
- **DER (Diarization Error Rate):** ~18.4% on DIHARD III, ~11.2% on AMI
- **Constraint:** `num_speakers=2` eliminates speaker count estimation errors

### NeMo ClusteringDiarizer + MSDD (Pipeline D)

- **VAD:** `vad_multilingual_marblenet` — 150ms window, 10ms shift, onset=0.4, offset=0.35
- **Embeddings:** `titanet_large` — multi-scale at 5 temporal resolutions:

| Window | Shift | Purpose |
|---|---|---|
| 1.5s | 0.75s | Long-range speaker characteristics |
| 1.25s | 0.625s | Medium-long context |
| 1.0s | 0.5s | Standard speaker turn |
| 0.75s | 0.375s | Short utterances |
| 0.5s | 0.1s | Brief interjections |

- **Clustering:** Agglomerative with MSDD refinement (`diar_msdd_telephony`)
- **MSDD:** Multi-Scale Diarization Decoder — neural network that fuses multi-scale embeddings for final speaker decisions
- **Config:** `enhanced_count_threshold=80`, `max_rp_threshold=0.25`, `sparse_search_volume=30`
- **DER:** ~12.8% on CALLHOME (telephony domain)

### DiariZen (Pipeline E)

- **Origin:** BUT-FIT team, winner of VoxSRC 2023 speaker diarization challenge
- **Approach:** End-to-end diarization (no separate VAD stage)
- **Fallback:** wespeaker CAM++ embeddings + spectral clustering
  - 1.5s sliding window, 0.5s hop
  - Agglomerative clustering (cosine distance, average linkage)
  - Segment merging for gaps < 0.5s
- **Strength:** Excellent on overlapping speech — trained specifically on challenge data with heavy overlap

---

## Separation Algorithms

### Spectral Leakage Suppression

Used for **all non-overlapping segments** across all pipelines. The core technique:

1. **Estimate leakage ratio** from solo speech segments (Stage 7)
2. **STFT** the primary and secondary channels
3. **Subtract** scaled secondary spectrum from primary: `|S_clean| = max(|S_prim| - 0.7 × leak × |S_sec|, 0.08 × |S_prim|)`
4. **Reconstruct** with original phase via inverse STFT

This removes cross-talk without introducing neural artifacts. The `floor=0.08` prevents the "musical noise" artifact common in spectral subtraction (silence shouldn't sound perfectly silent — a small amount of residual noise sounds more natural).

### Wiener Spectral Masking

Used for **overlap regions** in Pipelines B, D, E:

1. **Pre-compute** average magnitude spectrum per speaker from solo segments (Stage 8)
2. **STFT** the overlapping audio chunk
3. **Compute masks:** `mask_a[f] = |profile_a[f]|² / (|profile_a[f]|² + |profile_b[f]|²)`
4. **Apply:** `S_a = S_overlap × mask_a`, iSTFT to time domain

Wiener masking is optimal (in the MMSE sense) when the signal and noise are uncorrelated and the power spectra are known. Speaker profiles approximate these spectra.

### SepFormer Neural Separation

Used for **overlap regions** in Pipeline C:

- **Model:** Dual-path Transformer (`sepformer-whamr16k`)
- **Training data:** WHAM-R (Wall Street Journal + noise + reverberation) — better domain match than clean WSJ0-2mix
- **Architecture:** Encoder → dual-path Transformer (intra-chunk + inter-chunk attention) → Decoder → 2 masks
- **Published SI-SNRi:** 14.0 dB on WHAM-R test set
- **Permutation resolution:** Cosine similarity of separated sources against ECAPA-TDNN speaker centroids

---

## Transcription Engine

| Property | Value |
|---|---|
| **Model** | Whisper large-v3 (`faster-whisper` CTranslate2 backend) |
| **Compute** | float16 on GPU, float32 on CPU |
| **VAD** | Built-in Silero VAD, `min_silence_duration_ms=300` |
| **Word timestamps** | Enabled (for diarization alignment) |

**Diarization alignment process:**

1. Transcribe mixed audio with word-level timestamps
2. For each word, find the diarization segment that overlaps it
3. Assign the word to that segment's speaker label
4. Group consecutive same-speaker words into utterances
5. Start new utterance if gap > 0.5s between words
6. Split utterance if duration exceeds 20s

**Output format:**
```json
{
  "utterances": [
    { "speaker": "A", "start": 12.34, "end": 15.67, "text": "How long have you been experiencing this?" },
    { "speaker": "B", "start": 15.89, "end": 23.45, "text": "About two weeks now, it started with..." }
  ],
  "language": "en",
  "duration_s": 245.6,
  "num_segments": 47
}
```

---

## Evaluation Metrics

The system computes three standard speech quality metrics per pipeline output (`src/metrics.py`):

### SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

```
α = <estimated, reference> / <reference, reference>
SI-SDR = 10 × log10( ||α × reference||² / ||estimated - α × reference||² )
```

| Range | Interpretation |
|---|---|
| > 15 dB | Excellent separation |
| 10-15 dB | Good separation |
| 5-10 dB | Moderate — audible artifacts |
| < 5 dB | Poor — heavy distortion |

Scale-invariant: immune to overall gain differences. The primary metric for source separation quality.

### STOI (Short-Time Objective Intelligibility)

| Range | Interpretation |
|---|---|
| > 0.85 | Excellent intelligibility |
| 0.70-0.85 | Good intelligibility |
| 0.50-0.70 | Fair — some words lost |
| < 0.50 | Poor intelligibility |

Correlates strongly with human intelligibility scores (r > 0.9 in published studies). Computed on 25.6ms frames with one-third-octave band decomposition.

### PESQ (Perceptual Evaluation of Speech Quality)

| MOS-LQO | Interpretation |
|---|---|
| > 3.5 | Excellent (near-transparent) |
| 3.0-3.5 | Good |
| 2.5-3.0 | Fair |
| < 2.5 | Poor |

ITU-T P.862 standard. Perceptual model of the human auditory system. Range: 1.0 to 4.5 MOS-LQO.

### Leakage Attenuation

Measured during non-overlapping segments:

```
leakage_pct = energy_other_speaker / (energy_target_speaker + energy_other_speaker)
attenuation_dB = -20 × log10(mean_leakage_ratio)
```

Lower leakage percentage = better cross-talk suppression. Values below 5% indicate excellent isolation.

---

## API Reference

### Session Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/sessions` | List all sessions from `Audio_Files/` with metadata |
| `POST` | `/api/upload` | Upload WAV files (multipart form: `session_name` + `files`) |
| `GET` | `/api/health` | Check Modal deployment status + available pipelines |
| `GET` | `/api/pipelines` | List pipeline configurations and descriptions |

### Processing

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/process/{session_num}` | Launch all 5 pipelines. `?force=true` to reprocess |
| `GET` | `/api/process/{job_id}/status` | SSE stream: `{job_id, status, stage, pipeline_metadata}` |

**SSE event format:**
```
data: {"job_id": "abc123", "status": "running", "stage": "diarization", "pipeline_metadata": {...}}
```

### Results

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/results/{session_num}` | All pipeline outputs with metadata |
| `GET` | `/api/results/{session_num}/{pipeline}/{track}` | Serve WAV file (`pipeline`: energy/channel/sepformer/nemo/diarizen, `track`: speaker_a/speaker_b) |

**Result metadata per pipeline:**
```json
{
  "id": "sepformer",
  "elapsed_s": 94.2,
  "segments": 47,
  "overlap_pct": 18.3,
  "tracks": { "speaker_a": "/api/results/1/sepformer/speaker_a", ... },
  "error": null
}
```

### Transcription

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/transcribe/{session_num}` | Trigger Whisper transcription |
| `GET` | `/api/transcript/{session_num}` | Fetch transcript JSON |

---

## Frontend Architecture

### Tech Stack

| Package | Version | Purpose |
|---|---|---|
| React | 19.2.0 | UI framework |
| TypeScript | 5.9.3 | Type safety |
| Vite | 7.3.1 | Build tool + dev server (HMR, API proxy) |
| Tailwind CSS | 4.2.1 | Utility-first styling |
| WaveSurfer.js | 7.12.1 | Waveform visualization + audio playback |

### Component Hierarchy

```
App.tsx
├── UploadZone                    — drag-and-drop WAV upload
└── SessionCard[]                 — one per audio session
    ├── PipelineTabs              — switch between 5 pipeline results
    ├── WaveformPlayer (×2)       — Speaker A + Speaker B waveforms
    │   └── WaveSurfer instance   — interactive waveform, play/pause, seek, volume
    ├── TranscriptPanel           — diarized transcript with search, filter, auto-scroll
    │   └── Highlight             — search term highlighting in transcript
    └── ModelComparison           — side-by-side pipeline quality bars
```

### Design System

- **Palette:** Zinc-based neutral dark (`#09090b` base, `#18181b` cards, `#27272a` elevated)
- **Speaker colors:** A = `#818cf8` (indigo-400), B = `#34d399` (emerald-400)
- **Pipeline colors:** Energy `#71717a`, Channel `#60a5fa`, SepFormer `#a78bfa`, NeMo `#fbbf24`, DiariZen `#f87171`
- **Borders:** `rgba(255,255,255,0.06)` subtle
- **Animations:** Minimal — `fade-in` (200ms) and `slide-up` (300ms) only

---

## Modal GPU Deployment

### Container Image

```python
# Base: Debian slim + Python 3.10
# System: ffmpeg, libsndfile1, libglib2.0, git

# ML stack:
torch==2.1.2
torchaudio==2.1.2
pyannote.audio>=3.1.0,<3.4
speechbrain>=1.0.0,<1.1
nemo_toolkit[asr]==1.23.0
faster-whisper>=1.0.0

# Signal processing:
librosa>=0.10.1
scipy>=1.11.0,<2
scikit-learn>=1.3.0
noisereduce>=3.0.0

# Utilities:
soundfile>=0.12.1
transformers<4.38.0
peft==0.8.2
```

### Resource Configuration

| Parameter | Value | Rationale |
|---|---|---|
| GPU | A10G (24 GB VRAM) | Holds all 4 models simultaneously |
| Timeout | 2400s (40 min) | Longest pipeline (NeMo) on long recordings |
| Scaledown window | 300s (5 min) | Keeps container warm between sequential sessions |
| Volume | `speaker-sep-audio` | For files > 50 MB (avoids serialization overhead) |
| Secret | `huggingface-token` | pyannote model gating |

### Cold Start vs Warm Start

| Scenario | Time |
|---|---|
| First deploy (image build) | 5-10 minutes |
| Cold start (container spin-up) | ~15 seconds |
| Warm start (within 5 min window) | < 1 second |
| Model inference per session | 30-300 seconds (varies by pipeline) |

Models are loaded in `@modal.enter()` and persist across requests. The image caches all model weights (~3 GB total) so cold starts don't re-download.

---

## Configuration Reference

All parameters in `config.py`:

```python
# Modal
MODAL_APP_NAME          = "speaker-separation"
MODAL_GPU               = "A10G"           # 24 GB VRAM
MODAL_TIMEOUT           = 1800             # 30 min per session
MODAL_SCALEDOWN_WINDOW  = 300              # 5 min keep-warm
MODAL_VOLUME_NAME       = "speaker-sep-audio"
MODAL_SECRET_NAME       = "huggingface-token"

# Audio
TARGET_SAMPLE_RATE      = 16000            # Hz
AUDIO_NORM_PEAK_DB      = -1.0             # dB (output normalization)

# Denoising
DENOISE_ENABLED         = True
DENOISE_PROP_DECREASE   = 0.6              # 60% noise reduction
DENOISE_N_FFT           = 2048             # frequency resolution
DENOISE_FREQ_MASK_SMOOTH_HZ = 500
DENOISE_TIME_MASK_SMOOTH_MS  = 50

# Diarization
NUM_SPEAKERS            = 2

# Post-processing
MIN_SEGMENT_DURATION    = 0.25             # seconds
MAX_GAP_MERGE           = 0.3              # seconds (same-speaker gap merge)
MIN_ISLAND_DURATION     = 0.15             # seconds (remove isolated segments)
BOUNDARY_SNAP_MS        = 30               # ±ms (snap to energy minima)

# Embeddings
NUM_ENROLLMENT_SEGMENTS = 8                # per speaker
MIN_ENROLLMENT_DURATION = 0.5              # seconds

# Separation
FADE_DURATION_MS        = 20               # cosine fade at boundaries
MIN_SEPFORMER_DURATION  = 0.5              # seconds (minimum chunk)
SEPFORMER_PADDING       = 0.5              # seconds (zero-pad short chunks)
SEPFORMER_CONTEXT_S     = 0.25             # seconds (context around overlaps)

# Leakage suppression
LEAKAGE_ALPHA           = 0.7              # subtraction strength
LEAKAGE_FLOOR           = 0.08             # prevent musical noise

# Wiener filtering
WIENER_N_FFT            = 1024             # 64ms @ 16kHz
WIENER_HOP              = 256              # 16ms hop

# VAD gating
VAD_THRESHOLD_DB        = -40              # relative to track peak
VAD_ATTACK_MS           = 5                # fade-in
VAD_RELEASE_MS          = 80               # fade-out hold

# File transfer
VOLUME_THRESHOLD_BYTES  = 52428800         # 50 MB

# Speaker assignment
ASSIGNMENT_PRIMARY      = "first_speaker"
ASSIGNMENT_FALLBACK     = "channel_energy"

# Output
OUTPUT_SAMPLE_RATE      = 16000
OUTPUT_BIT_DEPTH        = 16               # PCM
OUTPUT_CHANNELS         = 1                # Mono
```

---

## Output Format

```
output/
├── session_001/
│   ├── energy/
│   │   ├── speaker_a.wav          # Energy-based separation
│   │   └── speaker_b.wav
│   ├── channel/
│   │   ├── speaker_a.wav          # Pyannote + Wiener
│   │   └── speaker_b.wav
│   ├── sepformer/
│   │   ├── speaker_a.wav          # Pyannote + SepFormer
│   │   └── speaker_b.wav
│   ├── nemo/
│   │   ├── speaker_a.wav          # NeMo MSDD
│   │   └── speaker_b.wav
│   ├── diarizen/
│   │   ├── speaker_a.wav          # DiariZen
│   │   └── speaker_b.wav
│   └── transcript.json            # Whisper large-v3 transcript
├── session_002/
│   └── ...
└── session_010/
    └── ...
```

All audio outputs: **16 kHz, mono, 16-bit PCM WAV, peak normalized to -1 dB.**

Per session: up to 10 WAV files (2 speakers x 5 pipelines) + 1 transcript JSON.

---

## Project Structure

```
├── README.md                        # This file
├── requirements.txt                 # Python dependencies (local)
├── config.py                        # All configurable parameters
├── server.py                        # FastAPI backend (port 8000)
├── main.py                          # CLI entry point
├── modal_app.py                     # Modal App + Image + GPU class
├── run_diarizen_all.py              # Batch: run Pipeline E on all sessions
│
├── src/
│   ├── __init__.py
│   ├── audio_utils.py               # Local audio I/O utilities
│   ├── pipeline.py                  # Pipelines B + C (pyannote + Wiener/SepFormer)
│   ├── energy_pipeline.py           # Pipeline A (energy-based, CPU)
│   ├── nemo_pipeline.py             # Pipeline D (NeMo MSDD + TitaNet)
│   ├── diarizen_pipeline.py         # Pipeline E (DiariZen + CAM++)
│   ├── transcription.py             # Whisper large-v3 transcription
│   └── metrics.py                   # SI-SDR, STOI, PESQ evaluation
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts               # Dev server + API proxy
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                   # Main layout + session management
│       ├── api.ts                    # REST + SSE client + TypeScript types
│       ├── index.css                 # Design tokens + dark theme
│       └── components/
│           ├── SessionCard.tsx       # Session display + processing controls
│           ├── UploadZone.tsx        # Drag-and-drop WAV upload
│           ├── WaveformPlayer.tsx    # WaveSurfer audio player
│           ├── PipelineTabs.tsx      # Pipeline result switcher
│           ├── ModelComparison.tsx   # Quality comparison bars
│           └── TranscriptPanel.tsx   # Diarized transcript viewer
│
├── Audio_Files/                     # Input: Session_001/ through Session_010/
├── output/                          # Output: per-pipeline separated audio
└── tests/
    └── test_pipeline.py             # Unit + integration tests
```

---

## Key Design Decisions

### Why downmix to mono

Two mics ~30 cm apart at 16 kHz sample rate: max frequency is 8 kHz (wavelength 4.3 cm). The mic spacing is ~7 wavelengths — deep in the spatial aliasing regime. Above 563 Hz (where spacing = half wavelength), there's no usable inter-channel difference. Both channels are near-identical copies with both speakers at similar levels. Beamforming, ICA, and spatial methods are ineffective. Mono simplifies all downstream processing.

### Why hybrid gate + separate

Running a neural separator (SepFormer) on the entire recording would introduce artifacts in the 70-85% where only one speaker is talking. By gating non-overlapping segments directly from the original channel with spectral leakage suppression, we get zero-artifact perfect quality for the majority of the recording. Neural separation is reserved only for the genuinely overlapping portions.

### Why SepFormer-WHAMR16K over WSJ02mix

WSJ0-2mix is clean, synthetically mixed speech. Clinical recordings have room reverberation, ambient noise, compression artifacts, and variable mic placement. The WHAMR16K variant was trained on noisy + reverberant mixtures — a much closer domain match to real-world conditions.

### Why embedding-based permutation resolution

SepFormer outputs `src1` and `src2` in arbitrary order — the mapping isn't consistent across chunks. Computing ECAPA-TDNN embeddings of each separated source and comparing to pre-computed speaker centroids via cosine similarity resolves this permutation problem reliably.

### Why 5 pipelines instead of 1

Different diarization models have different strengths:
- **pyannote** excels at overlap detection
- **NeMo MSDD** has multi-scale temporal analysis, strongest on telephony
- **DiariZen** was the VoxSRC 2023 winner for competitive diarization
- Recording conditions vary — no single model is best for all sessions

Running all 5 and comparing lets you pick the best result per session.

### Why SSE over WebSockets

Server-Sent Events are simpler for unidirectional updates (server → client). No handshake protocol, no keep-alive management, automatic reconnection built into the browser API. FastAPI's `StreamingResponse` handles it natively with zero additional dependencies.

### Why spectral subtraction over Wiener for non-overlaps

Non-overlapping segments have a known signal (target speaker) and a known interference (other speaker's leakage). Simple spectral subtraction with a measured leakage ratio is sufficient and introduces no estimation error from profile-based Wiener masks. Reserve Wiener masking for overlaps where both speakers are genuinely present.

---

## Challenges & Honest Assessment

- **Cross-talk from close mics is fundamentally hard.** Both speakers appear at similar levels on both channels (~50% leakage). Spatial/beamforming methods are ineffective at this mic spacing and sample rate. We rely on learned separation instead.

- **Real-world audio vs synthetic benchmarks.** SepFormer achieves 14.0 dB SI-SNRi on WHAM-R test, but clinical recordings with variable mic placement, room acoustics, and compression artifacts are harder. Expect 8-12 dB SI-SDR in practice.

- **Overlap separation has artifacts.** Non-overlapping segments (the majority) sound pristine since they're gated from the original. Overlapping regions will have audible but acceptable separation artifacts.

- **Speaker labeling is heuristic.** First-speaker and talk-time ratio rules work for typical clinical encounters but aren't guaranteed. Labels may need manual correction.

- **Diarization misses short utterances.** Brief backchannels ("mm-hmm", "okay") during the other speaker's turn may be misattributed or missed entirely.

- **Pipeline D (NeMo) is sensitive to config.** OmegaConf structured configs, `verbose` attribute propagation, and device placement require careful handling. Multiple iterations were needed to stabilize.

- **Pipeline E (DiariZen) has device mismatch risks.** The wespeaker fallback path requires forcing CPU mode to avoid CUDA/CPU tensor conflicts during embedding extraction.

---

## Cost Estimate

| Resource | Rate | Per Session (all 5 pipelines) | 10 Sessions |
|---|---|---|---|
| A10G GPU | ~$1.10/hr | ~$0.05-0.15 | ~$0.50-1.50 |
| Cold start | (first call only) | ~$0.02 | ~$0.02 |
| Transcription | (included in GPU time) | ~$0.01 | ~$0.10 |
| **Total** | | | **< $2.00** |

The container stays warm for 5 minutes between sessions — sequential batch processing avoids repeated cold starts. The $30 free Modal credit covers ~200+ sessions.

---

## Future Improvements

### Model & Architecture

1. **Fine-tune on clinical audio** — Domain-adapted diarization and separation models would significantly improve quality. Clinical speech has unique characteristics (medical terminology, specific conversational patterns, particular room acoustics) that general models don't capture.

2. **pyannote PixIT pipeline** — Joint diarization + separation in a single model. Eliminates the error propagation from separate diarize-then-separate stages. Could improve overlap handling substantially.

3. **MossFormer2 / ClearerVoice-Studio** — Newer Transformer-based separation architecture with published improvements over SepFormer on multiple benchmarks. Worth evaluating as a Pipeline C replacement.

4. **EEND-based diarization** — End-to-end neural diarization (e.g., EEND-EDA) directly outputs speaker labels per frame without clustering. Handles overlapping speech more naturally than cascaded systems.

5. **Speaker-conditioned separation** — Use enrollment audio of known speakers (e.g., the doctor's voice from previous sessions) for target speaker extraction instead of blind separation.

### Signal Processing

6. **Adaptive leakage estimation** — Currently uses a single global leakage ratio per speaker pair. Frequency-dependent leakage estimation would be more accurate (leakage characteristics vary across the spectrum).

7. **Phase-aware separation** — Current spectral subtraction preserves primary phase. Phase estimation (e.g., Griffin-Lim iterations or learned phase) could further improve quality.

8. **Bandwidth extension** — Output is currently 16 kHz. Super-resolution models could extend to 48 kHz for higher perceived quality when the application allows.

### System & UX

9. **Streaming/chunked processing** — Process long recordings in segments to reduce peak memory usage and enable incremental results. Currently the entire recording must fit in GPU memory.

10. **Confidence scoring per segment** — Flag low-confidence separations (high embedding ambiguity, low Wiener mask contrast) for human review. Surface these in the UI as yellow/red indicators.

11. **A/B listening tests** — Built-in interface for blind A/B comparison of pipeline outputs. Currently comparison is visual (metrics bars) — perceptual evaluation requires listening.

12. **Batch API** — REST endpoint to process multiple sessions in a single request with webhook callbacks. Currently requires one API call per session.

13. **Speaker enrollment** — Persistent speaker profiles across sessions. If the same doctor appears in 50 recordings, enrollment embeddings from confirmed sessions improve diarization accuracy.

14. **Export formats** — Support FLAC (lossless compression), MP3 (smaller files), SRT/VTT subtitles, and structured JSON with word-level alignments.

---

## Troubleshooting

**"Modal app not deployed"** — Run `modal deploy modal_app.py` first.

**"HF_TOKEN not found"** — Create the secret: `modal secret create huggingface-token HF_TOKEN=hf_xxxxx`

**"Access to model denied"** — Accept the pyannote model terms at the HuggingFace links in [Prerequisites](#prerequisites).

**First deploy is slow (~5-10 min)** — Normal. Modal builds the container image and downloads ~3 GB of ML models. Subsequent deploys use the cached image.

**Timeout on long recordings** — Default is 30 minutes. For recordings > 45 min, increase `MODAL_TIMEOUT` in `config.py` and redeploy.

**Output labels are swapped** — The speaker assignment is heuristic. Rename the output files if needed.

**NeMo "Key not in struct" error** — OmegaConf struct validation. Fixed by calling `OmegaConf.set_struct(cfg, False)` before ClusteringDiarizer instantiation.

**NeMo "verbose" attribute error** — Add `"verbose": True` to the top-level OmegaConf config dict. NeMo's ClusteringDiarizer reads `self.verbose` during VAD.

**DiariZen device mismatch** — wespeaker loads model weights on CUDA but `extract_embedding` creates CPU tensors. Force CPU with `set_gpu(-1)` if available, or use file-based extraction.

**Audio echo in output** — If overlap regions echo, check that the separation uses per-channel leakage suppression (not Wiener mixing of both channels).

**Frontend won't load** — Run `cd frontend && npm install && npm run build` first, or use `npm run dev` for development mode.

---

<p align="center">
  <sub>Built with pyannote.audio, SpeechBrain, NVIDIA NeMo, DiariZen, Modal, FastAPI, React, and WaveSurfer.js</sub>
</p>
