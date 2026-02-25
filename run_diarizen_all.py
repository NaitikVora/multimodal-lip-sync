"""Run DiariZen pipeline (E) for all 10 sessions via Modal GPU.

Usage: python run_diarizen_all.py
"""

import time
from pathlib import Path

import modal

from config import MODAL_APP_NAME
from src.audio_utils import (
    find_session_folder,
    load_session_audio_separate,
    save_wav_bytes,
)

AUDIO_DIR = Path("Audio_Files")
OUTPUT_DIR = Path("output")
SESSIONS = range(1, 11)


def run_diarizen_single(pipeline, session_num: int) -> dict:
    """Run just DiariZen for one session."""
    folder = find_session_folder(AUDIO_DIR, session_num)
    if folder is None:
        print(f"  [SKIP] Session {session_num} folder not found")
        return {}

    channels, info = load_session_audio_separate(folder)
    session_name = f"session_{session_num:03d}"
    print(f"  Audio: {info['num_files']} files, {info['duration_s']:.1f}s, "
          f"{info['total_bytes'] / (1024*1024):.1f} MB")

    # Call only run_pipeline_diarizen on Modal
    from src.diarizen_pipeline import run_pipeline_diarizen

    result = pipeline.process_diarizen_only.remote(
        channels["channel_a_bytes"],
        channels["channel_b_bytes"],
        channels["mixed_bytes"],
        session_name,
    )
    return result


def main():
    print("=" * 60)
    print("DiariZen Pipeline — All 10 Sessions")
    print("=" * 60)

    Cls = modal.Cls.from_name(MODAL_APP_NAME, "SpeakerSeparationPipeline")
    pipeline = Cls()

    total_t0 = time.time()

    for session_num in SESSIONS:
        session_name = f"session_{session_num:03d}"
        print(f"\n[Session {session_num:03d}] Starting...")
        t0 = time.time()

        folder = find_session_folder(AUDIO_DIR, session_num)
        if folder is None:
            print(f"  [SKIP] Folder not found")
            continue

        channels, info = load_session_audio_separate(folder)
        print(f"  Audio: {info['num_files']} files, {info['duration_s']:.1f}s, "
              f"{info['total_bytes'] / (1024*1024):.1f} MB")

        try:
            result = pipeline.process_diarizen_only.remote(
                channels["channel_a_bytes"],
                channels["channel_b_bytes"],
                channels["mixed_bytes"],
                session_name,
            )

            # Save results
            out_dir = OUTPUT_DIR / session_name / "diarizen"
            out_dir.mkdir(parents=True, exist_ok=True)

            meta = result.get("metadata", {})
            if meta.get("error"):
                print(f"  [ERROR] {meta['error']}")
            else:
                for track in ["speaker_a", "speaker_b"]:
                    if track in result and result[track]:
                        save_wav_bytes(result[track], out_dir / f"{track}.wav")
                elapsed = time.time() - t0
                print(f"  [OK] {meta.get('segments', '?')} segments, "
                      f"{meta.get('overlap_pct', '?')}% overlap, "
                      f"{elapsed:.1f}s total")

        except Exception as e:
            print(f"  [FAILED] {e}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"All done in {total_elapsed:.1f}s")
    print(f"Results saved to {OUTPUT_DIR}/session_XXX/diarizen/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
