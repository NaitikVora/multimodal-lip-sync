"""Model comparison metadata and scoring utilities.

Provides an objective quality scoring system for comparing pipeline outputs:
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - Leakage score (cross-talk attenuation)

These metrics are computed on the GPU container and returned as part
of the pipeline metadata, then displayed in the frontend comparison chart.
"""

import numpy as np


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Higher is better. Typical good values: >10 dB.
    Reference: Le Roux et al., 2019 (https://arxiv.org/abs/1811.02508)
    """
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    target = alpha * reference
    noise = estimate - target

    si_sdr = 10 * np.log10(
        (np.sum(target ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    )
    return float(si_sdr)


def compute_stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 16000,
) -> float:
    """Compute Short-Time Objective Intelligibility (STOI).

    Returns value in [0, 1], higher is better (>0.7 is good).
    Requires pystoi package.
    """
    try:
        from pystoi import stoi
        return float(stoi(reference, estimate, sr, extended=False))
    except ImportError:
        return -1.0


def compute_pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 16000,
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    Returns MOS-LQO score (1.0-4.5), higher is better (>3.0 is good).
    Requires pesq package.
    """
    try:
        from pesq import pesq as pesq_fn
        mode = "wb" if sr == 16000 else "nb"
        return float(pesq_fn(sr, reference[:len(estimate)], estimate[:len(reference)], mode))
    except ImportError:
        return -1.0
    except Exception:
        return -1.0


def compute_leakage_attenuation(
    speaker_a_track: np.ndarray,
    speaker_b_track: np.ndarray,
    non_overlap_segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    sr: int = 16000,
) -> dict:
    """Measure cross-talk attenuation.

    During Speaker A segments (speaker A speaking alone), measure how much
    of Speaker B's energy leaks into track A. Lower leakage → better separation.

    Returns:
        {"leakage_a_pct": float, "leakage_b_pct": float, "attenuation_db": float}
    """
    leakage_a_in_b = []  # A's energy in B's track during A-only segs
    leakage_b_in_a = []  # B's energy in A's track during B-only segs

    for seg in non_overlap_segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        if end <= start:
            continue

        dur = seg["end"] - seg["start"]
        if dur < 0.5:
            continue

        e_a_in_a = np.sqrt(np.mean(speaker_a_track[start:end] ** 2) + 1e-10)
        e_b_in_a = np.sqrt(np.mean(speaker_b_track[start:end] ** 2) + 1e-10)

        if seg["speaker"] == speaker_a_label:
            # A is speaking: A's track should be loud, B's track should be quiet
            leakage_b_in_a.append(e_b_in_a / (e_a_in_a + e_b_in_a + 1e-8))
        elif seg["speaker"] == speaker_b_label:
            leakage_a_in_b.append(e_a_in_a / (e_a_in_a + e_b_in_a + 1e-8))

    leak_a_pct = float(np.mean(leakage_a_in_b) * 100) if leakage_a_in_b else 50.0
    leak_b_pct = float(np.mean(leakage_b_in_a) * 100) if leakage_b_in_a else 50.0

    # Attenuation in dB: how much quieter is the leakage vs signal
    avg_leak = (leak_a_pct + leak_b_pct) / 200  # scale to [0,1]
    attenuation_db = -20 * np.log10(avg_leak + 1e-8) if avg_leak > 0 else 0.0

    return {
        "leakage_a_pct": round(leak_a_pct, 1),
        "leakage_b_pct": round(leak_b_pct, 1),
        "attenuation_db": round(float(attenuation_db), 1),
    }


def score_pipeline(
    speaker_a_track: np.ndarray,
    speaker_b_track: np.ndarray,
    mixed_audio: np.ndarray,
    non_overlap_segments: list[dict],
    speaker_a_label: str,
    speaker_b_label: str,
    sr: int = 16000,
) -> dict:
    """Compute all quality scores for a pipeline's output.

    Returns a dict suitable for inclusion in pipeline metadata.
    """
    leakage = compute_leakage_attenuation(
        speaker_a_track, speaker_b_track, non_overlap_segments,
        speaker_a_label, speaker_b_label, sr,
    )

    # SI-SDR against the mixed signal (self-consistency measure)
    si_sdr_a = compute_si_sdr(mixed_audio[:len(speaker_a_track)], speaker_a_track)
    si_sdr_b = compute_si_sdr(mixed_audio[:len(speaker_b_track)], speaker_b_track)

    # STOI (may return -1 if not installed)
    stoi_a = compute_stoi(mixed_audio[:len(speaker_a_track)], speaker_a_track, sr)
    stoi_b = compute_stoi(mixed_audio[:len(speaker_b_track)], speaker_b_track, sr)

    return {
        "si_sdr_a": round(si_sdr_a, 2),
        "si_sdr_b": round(si_sdr_b, 2),
        "stoi_a": round(stoi_a, 3) if stoi_a >= 0 else None,
        "stoi_b": round(stoi_b, 3) if stoi_b >= 0 else None,
        **leakage,
    }
