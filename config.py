"""Configurable parameters for the speaker separation pipeline."""

# --- Modal Configuration ---
MODAL_APP_NAME = "speaker-separation"
MODAL_GPU = "A10G"
MODAL_TIMEOUT = 1800  # 30 min per session
MODAL_SCALEDOWN_WINDOW = 300  # Keep warm 5 min between sessions
MODAL_VOLUME_NAME = "speaker-sep-audio"
MODAL_SECRET_NAME = "huggingface-token"

# --- Audio Configuration ---
TARGET_SAMPLE_RATE = 16000
AUDIO_NORM_PEAK_DB = -1.0  # Peak normalization target (dB)

# --- Denoising ---
DENOISE_ENABLED = True
DENOISE_PROP_DECREASE = 0.6  # Conservative to avoid speech distortion

# --- Diarization ---
NUM_SPEAKERS = 2

# --- Speaker Embeddings ---
NUM_ENROLLMENT_SEGMENTS = 5  # Top N longest non-overlap segments per speaker
MIN_ENROLLMENT_DURATION = 1.0  # Minimum segment length (seconds) for enrollment

# --- Separation ---
FADE_DURATION_MS = 15  # Cosine fade at segment boundaries (milliseconds)
MIN_SEPFORMER_DURATION = 0.5  # Min chunk length for SepFormer (seconds)
SEPFORMER_PADDING = 0.5  # Zero-pad short chunks to this length (seconds)

# --- File Transfer ---
VOLUME_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB — switch to Volume transfer above this

# --- Doctor/Patient Assignment ---
# "first_speaker" — whoever speaks first is labeled doctor
# "less_speaking" — speaker with less total time is doctor
ASSIGNMENT_PRIMARY = "first_speaker"
ASSIGNMENT_FALLBACK = "less_speaking"

# --- Output ---
OUTPUT_SAMPLE_RATE = 16000
OUTPUT_BIT_DEPTH = 16  # 16-bit PCM
OUTPUT_CHANNELS = 1  # Mono
