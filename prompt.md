# Claude Code Prompt: Speaker Separation Tool using Modal (GPU Cloud) for Mac

## PROJECT OVERVIEW

Build a Python-based speaker separation tool that takes messy clinical audio recordings (doctor + patient conversations with heavy cross-talk from closely-spaced microphones) and produces clean, speaker-separated output files. All heavy GPU computation runs on **Modal** (serverless GPU cloud). The local Mac machine handles only CLI orchestration, file upload/download, and lightweight tasks.

This is a technical interview assignment for Luel.

## CONTEXT & CONSTRAINTS

- **Developer machine**: macOS (no NVIDIA GPU, no CUDA)
- **Compute**: Modal serverless GPU (A10G or T4, 24GB / 16GB VRAM)
- **Audio scenario**: Two speakers (doctor + patient), two microphones ~1 foot apart, heavy cross-talk (both voices on both channels at similar levels), muddy quality, room tone, compression artifacts
- **Input**: 10 sessions of audio files (WAV format, stereo or multi-channel), stored in a local `Audio_Files/` directory with session subfolders (Session 1 through Session 10)
- **Output per session (30 files total)**:
  - `session_N_original.wav` — raw/original recording
  - `session_N_doctor.wav` — doctor's voice only (other speaker suppressed)
  - `session_N_patient.wav` — patient's voice only (other speaker suppressed)
- **Language**: Python 3.10+
- **Key requirement**: The user runs everything from their Mac terminal. Modal handles all GPU work transparently.

## HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│  LOCAL MAC                                           │
│                                                      │
│  main.py (CLI entry point)                           │
│    ├── Reads audio files from Audio_Files/           │
│    ├── Uploads audio bytes to Modal Volume           │
│    ├── Calls Modal GPU functions remotely             │
│    ├── Downloads results from Modal Volume            │
│    └── Saves output/ directory locally                │
└──────────────────────┬──────────────────────────────┘
                       │ modal.Function.remote()
                       ▼
┌─────────────────────────────────────────────────────┐
│  MODAL CLOUD (A10G GPU)                              │
│                                                      │
│  SpeakerSeparationPipeline (modal.Cls)              │
│    ├── @enter: Load all models once into GPU memory  │
│    │   ├── pyannote diarization 3.1                  │
│    │   ├── SpeechBrain ECAPA-TDNN embeddings         │
│    │   └── SpeechBrain SepFormer (whamr16k)          │
│    │                                                  │
│    ├── @method process_session()                     │
│    │   ├── Stage 0: Load audio, downmix, resample    │
│    │   ├── Stage 1: Denoise (noisereduce)            │
│    │   ├── Stage 2: Diarize (pyannote, num_speakers=2)│
│    │   ├── Stage 3: Extract speaker embeddings       │
│    │   ├── Stage 4: Hybrid separation                │
│    │   │   ├── Non-overlap: gate original audio      │
│    │   │   └── Overlap: SepFormer + embedding match  │
│    │   ├── Stage 5: Stitch & post-process            │
│    │   └── Stage 6: Assign doctor vs patient         │
│    │                                                  │
│    └── Returns: dict of WAV bytes for each output    │
│                                                      │
│  Volume: speaker-sep-data (upload/download audio)    │
└─────────────────────────────────────────────────────┘
```

## PROJECT STRUCTURE

```
speaker-separation/
├── README.md                        # Setup, usage, approach write-up
├── requirements.txt                 # LOCAL dependencies only (modal, soundfile, rich)
├── config.py                        # All configurable parameters
├── main.py                          # LOCAL CLI entry point — orchestrates everything
├── modal_app.py                     # MODAL — defines App, Image, Cls, all GPU functions
├── src/
│   ├── __init__.py
│   ├── audio_utils.py               # LOCAL — audio loading, format detection, basic I/O
│   └── pipeline.py                  # MODAL (runs inside container) — full separation pipeline
├── output/                          # Where the 30 output files go locally
└── tests/
    └── test_pipeline.py
```

## MODAL SETUP — CRITICAL DETAILS

### Modal Image Definition (in modal_app.py)

```python
import modal

app = modal.App("speaker-separation")

# Define the GPU container image with ALL dependencies baked in
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "pyannote.audio>=3.1.0",
        "speechbrain>=1.0.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "noisereduce>=3.0.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "rich>=13.0.0",
    )
)

# Volume for transferring audio files between local and cloud
audio_volume = modal.Volume.from_name("speaker-sep-audio", create_if_missing=True)

# HuggingFace token for pyannote model access
hf_secret = modal.Secret.from_name("huggingface-token")
# User must create this secret via: modal secret create huggingface-token HF_TOKEN=hf_xxxxx
```

### Modal Class for GPU Processing (in modal_app.py)

```python
@app.cls(
    image=gpu_image,
    gpu="A10G",  # 24GB VRAM — enough for all models simultaneously
    timeout=1800,  # 30 min timeout per session (clinical recordings can be long)
    volumes={"/data": audio_volume},
    secrets=[hf_secret],
    container_idle_timeout=300,  # Keep warm for 5 min between sessions
)
class SpeakerSeparationPipeline:
    @modal.enter()
    def load_models(self):
        """Load all models into GPU memory ONCE when container starts.
        This runs only on first call — subsequent calls reuse loaded models."""
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Diarization pipeline
        self.diarizer = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ["HF_TOKEN"]
        )
        self.diarizer.to(self.device)

        # 2. Speaker embedding model
        self.embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)}
        )

        # 3. Source separation model (noisy+reverberant variant)
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            run_opts={"device": str(self.device)}
        )

        print("All models loaded successfully!")

    @modal.method()
    def process_session(self, audio_bytes: bytes, session_name: str) -> dict:
        """Process a single session. Receives raw WAV bytes, returns dict of output WAV bytes."""
        # ... full pipeline implementation here ...
        return {
            "original": original_wav_bytes,
            "doctor": doctor_wav_bytes,
            "patient": patient_wav_bytes,
        }
```

### Local Entrypoint (in main.py)

```python
# main.py runs on your Mac — NO GPU needed locally
import modal
from pathlib import Path

# Import the remote Modal app
app = modal.App.lookup("speaker-separation")
Pipeline = modal.Cls.from_name("speaker-separation", "SpeakerSeparationPipeline")

def main(input_dir: str, output_dir: str):
    pipeline = Pipeline()

    for session_num in range(1, 11):
        session_path = find_session_audio(input_dir, session_num)
        audio_bytes = session_path.read_bytes()

        # This call runs on Modal's GPU cloud!
        results = pipeline.process_session.remote(audio_bytes, f"session_{session_num}")

        # Save results locally
        for name, wav_bytes in results.items():
            output_path = Path(output_dir) / f"session_{session_num}_{name}.wav"
            output_path.write_bytes(wav_bytes)
            print(f"Saved: {output_path}")
```

## PIPELINE IMPLEMENTATION — DETAILED STAGES (runs inside Modal container)

### Stage 0: Audio Loading & Preprocessing
- Receive raw WAV bytes as input (already uploaded via function argument)
- Use `soundfile` to read from `io.BytesIO(audio_bytes)`
- If stereo/multi-channel, downmix to mono (the 2 mics at 1-foot spacing provide negligible spatial diversity — explain this in README)
- Resample to 16kHz using `librosa.resample()` (required by all models)
- Normalize audio levels to [-1, 1] range
- Keep the original audio bytes for the `_original.wav` output

### Stage 1: Speech Enhancement / Denoising
- Use `noisereduce` library (pure Python, no complex dependencies) to reduce room tone and stationary noise
- Apply `noisereduce.reduce_noise(y=audio, sr=16000, prop_decrease=0.6)` — conservative to avoid speech distortion
- This cleaned audio feeds into diarization and separation stages
- If noisereduce causes issues, make it skippable via config flag

### Stage 2: Speaker Diarization (Who spoke when)
- Use `pyannote.audio` pipeline `pyannote/speaker-diarization-3.1`
- **Critical**: Set `num_speakers=2` to constrain to exactly doctor + patient
- Input: the denoised mono 16kHz audio as a temporary WAV file (pyannote needs a file path or `{"waveform": tensor, "sample_rate": int}` dict)
- Output: `pyannote.core.Annotation` object with timeline of speaker segments
- Extract as list of tuples: `[(speaker_label, start_sec, end_sec), ...]`
- Also identify **overlap regions** where both speakers are marked as active

### Stage 3: Speaker Embedding Extraction
- From diarization output, find the **longest non-overlapping segments** for each speaker
- Use top 3-5 segments per speaker for robust embedding
- Extract embeddings using `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN)
- For each segment: load audio slice → pass through embedder → get 192-dim embedding vector
- Average embeddings per speaker to get a robust centroid
- Store as `{speaker_label: embedding_tensor}`

### Stage 4: Audio Separation — Hybrid Approach

#### For NON-OVERLAPPING segments (~70-85% of typical conversation):
- Simply **extract/gate** the original (denoised) audio using diarization timestamps
- For each segment belonging to Speaker A: copy audio from `start` to `end` into Speaker A's track, put silence in Speaker B's track
- Apply soft cosine fade-in/fade-out (15ms) at segment boundaries to avoid clicks
- **This preserves original audio quality with zero separation artifacts** — this is the key insight

#### For OVERLAPPING segments:
- Extract the overlapping audio chunk
- Run through **SepFormer** (`speechbrain/sepformer-whamr16k`) to get 2 separated sources
- **Solve the permutation problem**: For each separated source, extract speaker embedding and compute cosine similarity against the enrollment embeddings from Stage 3
- Assign each separated source to the speaker with highest similarity
- Place each assigned source into the correct speaker track
- If SepFormer chunk is short (<0.5s), apply zero-padding before separation and trim after

### Stage 5: Stitching & Post-processing
- Both speaker tracks should be the **same length** as the original audio
- Fill silence periods with zeros (when a speaker is not talking)
- Normalize each track to -1dB peak (avoid clipping)
- Convert to 16-bit PCM WAV at 16kHz mono
- Serialize to bytes using `soundfile.write()` into `io.BytesIO()`

### Stage 6: Doctor vs Patient Assignment
- **Heuristic**: The speaker who speaks first is labeled "doctor" (doctors typically initiate clinical encounters with greetings/introductions)
- **Fallback heuristic**: If speaking times are very unequal, the speaker with less total speaking time is likely the doctor (patients tend to have longer symptom descriptions)
- **Important**: Document in README that this is a heuristic and the user can swap labels if incorrect for their specific recordings
- In the output dict, return `"doctor"` and `"patient"` keys based on this assignment

## FILE TRANSFER STRATEGY

There are two approaches. Choose the simpler one:

**Option A (Preferred — Direct byte passing):**
- Read audio file as bytes locally
- Pass bytes directly as argument to `pipeline.process_session.remote(audio_bytes, session_name)`
- Receive results dict with WAV bytes back
- Write bytes to local files
- Pros: Simple, no Volume management needed for data transfer
- Cons: Large files may be slow to serialize; Modal has ~50MB limit for function arguments

**Option B (For large files — Use Volume):**
- Upload audio files to Modal Volume before processing
- Modal function reads from Volume, writes results to Volume
- Download results from Volume after processing
- Use `volume.batch_upload()` and `modal volume get` commands
- Necessary if individual session audio files exceed ~50MB

**Implement Option A first.** If audio files are too large, fall back to Option B with Volume. Implement both and auto-detect based on file size.

## LOCAL REQUIREMENTS (requirements.txt for Mac)

```
modal>=0.64.0
soundfile>=0.12.1
librosa>=0.10.1
rich>=13.0.0
click>=8.1.0
```

Note: Do NOT include torch, pyannote, speechbrain in local requirements — those only run in the Modal container.

## CLI INTERFACE (main.py on Mac)

```bash
# First-time setup
pip install -r requirements.txt
modal setup  # Authenticate with Modal
modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxx

# Deploy the Modal app (builds the container image with all ML models)
modal deploy modal_app.py

# Process all 10 sessions
python main.py --input-dir ./Audio_Files --output-dir ./output

# Process a single session
python main.py --input-dir ./Audio_Files --output-dir ./output --session 3

# Dry run (just check files, don't process)
python main.py --input-dir ./Audio_Files --output-dir ./output --dry-run
```

Use `click` for CLI argument parsing. Use `rich` for progress display and nice formatting.

## ERROR HANDLING & EDGE CASES

1. **HuggingFace token missing**: If `modal secret` for `huggingface-token` isn't created, provide clear instructions. pyannote/speaker-diarization-3.1 requires the user to accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1 AND have an HF token.

2. **Large audio files**: If a session's audio exceeds 50MB, use Volume-based transfer instead of direct byte passing. Print a message explaining why.

3. **Audio format detection**: Sessions may contain multiple WAV files (e.g., one per channel). If a session folder has multiple WAVs, load them all and mix down. If it has a single stereo WAV, downmix to mono. Handle both cases.

4. **Model download on first run**: The first `modal deploy` or `modal run` will take 5-10 minutes as it builds the container image and downloads models. Print a message explaining this.

5. **Timeout**: Clinical recordings can be 30-60+ minutes. Set Modal function timeout to 1800s (30 min). For very long recordings, process in chunks.

6. **Graceful degradation**:
   - If pyannote fails: Fall back to simple energy-based VAD + spectral clustering
   - If SepFormer fails on overlap segments: Just mix overlap equally between speakers
   - If embedding matching fails: Use diarization labels directly

7. **Session folder structure**: The Audio_Files directory might have varying structures (e.g., `Session 1/`, `Session1/`, numbered subfolders). Be flexible in detecting session folders.

## README.md MUST INCLUDE

1. **Quick Start**: 
   ```
   1. pip install -r requirements.txt
   2. modal setup
   3. modal secret create huggingface-token HF_TOKEN=hf_xxxxx
   4. modal deploy modal_app.py
   5. python main.py --input-dir ./Audio_Files --output-dir ./output
   ```

2. **Prerequisites**: Modal account ($30 free credit), HuggingFace account + token (accept pyannote model terms)

3. **How it works**: Clear pipeline diagram (ASCII) showing local Mac ↔ Modal cloud architecture

4. **Why Modal**: Explain that the ML models require GPU (CUDA), Modal provides on-demand A10G GPUs at ~$1.10/hr, and the developer's Mac doesn't have NVIDIA GPU

5. **Pipeline explanation**: Each stage described with reasoning

6. **Libraries used and why**: 
   - pyannote.audio: Best-in-class diarization with native overlap detection
   - SpeechBrain SepFormer: State-of-the-art separation with noisy/reverberant pretrained model
   - ECAPA-TDNN: Robust speaker embeddings for permutation resolution
   - noisereduce: Lightweight denoising without heavy dependencies
   - Modal: Serverless GPU access from Mac

7. **Key design decisions**:
   - Why downmix to mono (1-foot mic spacing → negligible spatial diversity above 563Hz)
   - Why hybrid gate+separate (preserve quality on non-overlaps, separate only where needed)
   - Why sepformer-whamr16k over wsj02mix (trained on noisy+reverberant → better domain match)
   - Why embedding-based permutation resolution (SepFormer output order is arbitrary)

8. **Challenges & honest assessment**:
   - Real-world audio is much harder than synthetic benchmarks
   - Separation quality on overlapping segments won't be studio-clean
   - Doctor/patient assignment is heuristic-based
   - Cross-talk from close mics is fundamentally hard

9. **What I'd improve with more time**:
   - Fine-tune models on clinical audio domain
   - Try pyannote PixIT joint diarization-separation pipeline
   - Use ClearerVoice-Studio MossFormer2 for potentially better separation
   - Add a web UI for easier interaction
   - Implement confidence scores per segment

10. **Cost estimate**: ~$0.02-0.05 per session on Modal (A10G at $1.10/hr, ~1-3 min per session)

## IMPLEMENTATION PRIORITIES

1. **First**: Get the Modal infrastructure working — image builds, models load, a simple test audio processes end-to-end
2. **Second**: Implement the full pipeline with real diarization + gating (non-overlap only)
3. **Third**: Add SepFormer separation for overlap regions + embedding matching
4. **Fourth**: Polish CLI, error handling, progress bars
5. **Fifth**: Process all 10 sessions, verify output quality
6. **Last**: Write comprehensive README

## QUALITY CHECKLIST

Before considering the implementation complete:
- [ ] `modal deploy modal_app.py` succeeds and builds container image
- [ ] All 10 sessions process without errors via `python main.py`
- [ ] Each session produces exactly 3 output files (original, doctor, patient)
- [ ] Output audio is audible and reasonable (not just noise or silence)
- [ ] Doctor and patient tracks contain clearly different speakers
- [ ] Silence periods are properly handled (silence when speaker isn't talking)
- [ ] No audio clipping in output files
- [ ] README is complete with setup instructions that work on Mac
- [ ] Code is clean, commented, and follows Python conventions
- [ ] requirements.txt includes ONLY local dependencies
- [ ] modal_app.py has all cloud dependencies in the Image definition
- [ ] The tool can process a new audio file (not just the test data)
- [ ] Total Modal cost for all 10 sessions is under $1

## WHAT SUCCESS LOOKS LIKE

A user on a Mac should be able to:
1. Clone the repo
2. `pip install -r requirements.txt`
3. `modal setup` + create HF secret
4. `modal deploy modal_app.py` (wait for image build on first run)
5. `python main.py --input-dir ./Audio_Files --output-dir ./output`
6. Get 30 WAV files in `./output/` with recognizable speaker separation
7. Total wall-clock time: ~15-30 minutes for all 10 sessions (most is GPU processing)
8. Total Modal cost: under $1

## NOW BUILD IT

Start by creating modal_app.py with the Image and Cls definitions. Verify the image builds with `modal deploy`. Then implement the pipeline stages one at a time inside the Modal Cls. Test with a single short audio file before running all 10 sessions. The local main.py should be the last thing you build — get the cloud processing working first.
