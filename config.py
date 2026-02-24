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
DENOISE_N_FFT = 2048  # Better frequency resolution than default 1024
DENOISE_FREQ_MASK_SMOOTH_HZ = 500  # Smoother frequency masking
DENOISE_TIME_MASK_SMOOTH_MS = 50  # Smoother time masking

# --- Diarization ---
NUM_SPEAKERS = 2

# --- Diarization Post-Processing ---
MIN_SEGMENT_DURATION = 0.25  # Merge segments shorter than this (seconds)
MAX_GAP_MERGE = 0.3  # Merge same-speaker segments with gaps under this (seconds)
MIN_ISLAND_DURATION = 0.15  # Remove isolated segments shorter than this (seconds)
BOUNDARY_SNAP_MS = 30  # Snap boundaries to energy valleys within this window (ms)

# --- Speaker Embeddings ---
NUM_ENROLLMENT_SEGMENTS = 8  # Top N longest non-overlap segments per speaker
MIN_ENROLLMENT_DURATION = 0.5  # Minimum segment length (seconds) for enrollment

# --- Separation ---
FADE_DURATION_MS = 20  # Cosine fade at segment boundaries (milliseconds)
MIN_SEPFORMER_DURATION = 0.5  # Min chunk length for SepFormer (seconds)
SEPFORMER_PADDING = 0.5  # Zero-pad short chunks to this length (seconds)
SEPFORMER_CONTEXT_S = 0.25  # Context padding around overlap chunks for SepFormer

# --- Cross-channel leakage suppression ---
LEAKAGE_ALPHA = 0.7  # Spectral subtraction strength (0-1, lower = safer)
LEAKAGE_FLOOR = 0.08  # Spectral floor to prevent musical noise artifacts

# --- Wiener overlap separation ---
WIENER_N_FFT = 1024  # FFT size for spectral masking (64ms @ 16kHz)
WIENER_HOP = 256  # Hop size for STFT (16ms)

# --- VAD gating (post-separation cleanup) ---
VAD_THRESHOLD_DB = -40  # Gate threshold relative to track peak
VAD_ATTACK_MS = 5  # Fade-in time when speech starts
VAD_RELEASE_MS = 80  # Fade-out hold time after speech ends

# --- File Transfer ---
VOLUME_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB — switch to Volume transfer above this

# --- Speaker Assignment ---
# "first_speaker" — whoever speaks first is labeled Speaker A
# "channel_energy" — use channel energy correlation to assign
ASSIGNMENT_PRIMARY = "first_speaker"
ASSIGNMENT_FALLBACK = "channel_energy"

# --- Output ---
OUTPUT_SAMPLE_RATE = 16000
OUTPUT_BIT_DEPTH = 16  # 16-bit PCM
OUTPUT_CHANNELS = 1  # Mono
