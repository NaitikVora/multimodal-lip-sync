"""Local CLI entry point — runs on Mac, orchestrates Modal GPU processing.

Usage:
    python main.py --input-dir ./Audio_Files --output-dir ./output
    python main.py --input-dir ./Audio_Files --output-dir ./output --session 3
    python main.py --input-dir ./Audio_Files --output-dir ./output --dry-run
"""

import sys
import time
from pathlib import Path

import click
import modal
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import MODAL_APP_NAME, VOLUME_THRESHOLD_BYTES
from src.audio_utils import (
    discover_sessions,
    find_session_folder,
    load_session_audio,
    save_wav_bytes,
)

console = Console()


def get_pipeline(use_volume: bool = False):
    """Look up the deployed Modal pipeline class."""
    cls_name = (
        "SpeakerSeparationPipelineVolume" if use_volume
        else "SpeakerSeparationPipeline"
    )
    try:
        Pipeline = modal.Cls.from_name(MODAL_APP_NAME, cls_name)
        return Pipeline()
    except modal.exception.NotFoundError:
        console.print(
            "\n[bold red]Error:[/] Modal app not deployed. Run first:\n"
            "  [cyan]modal deploy modal_app.py[/]\n"
        )
        sys.exit(1)


def process_session_direct(pipeline, audio_bytes: bytes, session_name: str) -> dict:
    """Send audio bytes directly to Modal and get results back."""
    return pipeline.process_session.remote(audio_bytes, session_name)


def process_session_volume(pipeline_vol, audio_bytes: bytes, session_name: str) -> dict:
    """Upload to Volume, process, download results."""
    upload_fn = modal.Function.from_name(MODAL_APP_NAME, "upload_to_volume")
    download_fn = modal.Function.from_name(MODAL_APP_NAME, "download_from_volume")

    input_path = f"input/{session_name}.wav"
    upload_fn.remote(audio_bytes, input_path)

    output_prefix = f"output/{session_name}"
    output_paths = pipeline_vol.process_session_volume.remote(
        input_path, output_prefix, session_name
    )

    results = {}
    for key, remote_path in output_paths.items():
        results[key] = download_fn.remote(remote_path)

    return results


@click.command()
@click.option(
    "--input-dir", "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing session subfolders with WAV files.",
)
@click.option(
    "--output-dir", "-o",
    default="./output",
    type=click.Path(file_okay=False),
    help="Directory for output WAV files.",
)
@click.option(
    "--session", "-s",
    type=int,
    default=None,
    help="Process only this session number.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Scan files only, don't process.",
)
@click.option(
    "--health-check",
    is_flag=True,
    help="Check that the Modal app is deployed and models are loaded.",
)
def main(input_dir: str, output_dir: str, session: int, dry_run: bool, health_check: bool):
    """Speaker Separation Tool — processes clinical audio via Modal GPU cloud."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    console.print(Panel.fit(
        "[bold]Speaker Separation Tool[/]\n"
        "Clinical audio → separated doctor & patient tracks\n"
        f"Input:  {input_path.resolve()}\n"
        f"Output: {output_path.resolve()}",
        border_style="blue",
    ))

    # Health check mode
    if health_check:
        console.print("\n[bold]Running health check...[/]")
        try:
            pipeline = get_pipeline()
            status = pipeline.health_check.remote()
            console.print(f"[green]✓[/] {status}")
        except Exception as e:
            console.print(f"[red]✗[/] Health check failed: {e}")
            sys.exit(1)
        return

    # Discover sessions
    if session is not None:
        session_nums = [session]
    else:
        session_nums = discover_sessions(input_path)

    if not session_nums:
        console.print("[red]No session folders found.[/]")
        sys.exit(1)

    # Build session info table
    table = Table(title="Sessions Found")
    table.add_column("Session", style="cyan")
    table.add_column("Folder", style="white")
    table.add_column("Files", style="white")
    table.add_column("Total Size", style="yellow")

    sessions_to_process = []
    for num in session_nums:
        folder = find_session_folder(input_path, num)
        if folder is None:
            table.add_row(str(num), "[red]NOT FOUND[/]", "-", "-")
            continue

        wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
        total_size = sum(f.stat().st_size for f in wav_files)
        size_mb = total_size / (1024 * 1024)

        table.add_row(
            str(num),
            folder.name,
            str(len(wav_files)),
            f"{size_mb:.1f} MB",
        )
        sessions_to_process.append((num, folder, total_size))

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run — no processing performed.[/]")
        return

    if not sessions_to_process:
        console.print("[red]No valid sessions to process.[/]")
        sys.exit(1)

    # Process sessions
    output_path.mkdir(parents=True, exist_ok=True)
    total_start = time.time()
    success_count = 0
    fail_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        overall_task = progress.add_task(
            f"Processing {len(sessions_to_process)} sessions...",
            total=len(sessions_to_process),
        )

        for num, folder, total_size in sessions_to_process:
            session_name = f"session_{num:03d}"
            progress.update(overall_task, description=f"Processing {session_name}...")

            try:
                # Load and mix audio locally
                audio_bytes, info = load_session_audio(folder)
                console.print(
                    f"\n[bold cyan]{session_name}[/]: "
                    f"{info['num_files']} files, "
                    f"{info['mixed_duration_s']:.1f}s, "
                    f"{len(audio_bytes)/1024/1024:.1f} MB mixed"
                )

                # Choose transfer method based on size
                use_volume = len(audio_bytes) > VOLUME_THRESHOLD_BYTES
                if use_volume:
                    console.print(
                        f"  [yellow]File > 50 MB — using Volume-based transfer[/]"
                    )
                    pipeline = get_pipeline(use_volume=True)
                    results = process_session_volume(pipeline, audio_bytes, session_name)
                else:
                    pipeline = get_pipeline(use_volume=False)
                    results = process_session_direct(pipeline, audio_bytes, session_name)

                # Save output files
                for key, wav_bytes in results.items():
                    out_file = output_path / f"{session_name}_{key}.wav"
                    save_wav_bytes(wav_bytes, out_file)
                    console.print(f"  [green]✓[/] Saved: {out_file.name} ({len(wav_bytes)/1024:.0f} KB)")

                success_count += 1

            except Exception as e:
                console.print(f"  [red]✗[/] Failed: {e}")
                fail_count += 1

            progress.advance(overall_task)

    elapsed = time.time() - total_start
    console.print(Panel.fit(
        f"[bold green]Done![/]\n"
        f"Processed: {success_count}/{len(sessions_to_process)} sessions\n"
        f"Failed: {fail_count}\n"
        f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n"
        f"Output: {output_path.resolve()}",
        border_style="green" if fail_count == 0 else "yellow",
    ))


if __name__ == "__main__":
    main()
