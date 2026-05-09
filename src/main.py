from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from chunker import chunk_text
from converter import convert_wavs_to_mp3, save_wav
from hardware import get_device
from inference import create_prompt, generate_chunks, load_model

# ANSI color codes
DIM    = "\033[2m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .txt book into MP3 audiobook using OmniVoice TTS."
    )
    parser.add_argument("--file", required=True, help="Path to input .txt file")
    parser.add_argument(
        "--device",
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "sk", "cs"],
        help="Language (default: en)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice reference sample: filename in voices/ or full path",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help='Voice design attributes, e.g. "female, young adult, high pitch"',
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10240,
        help="Chunk size in bytes (default: 10240)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Batch size for TTS / parallel ffmpeg workers (default: CPU count)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=None,
        help="Start (or resume) from this chunk number, 1-based inclusive",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="Diffusion steps per chunk (default: 32, faster: 16)",
    )
    parser.add_argument(
        "--tts-batch",
        type=int,
        default=4,
        help="Number of chunks per GPU batch (default: 4)",
    )
    return parser.parse_args()


def get_completed_indices(output_dir: str) -> set[int]:
    """Scan output directory for already completed chunks."""
    completed = set()
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed

    for f in output_path.iterdir():
        if f.suffix in (".wav", ".mp3") and f.stem.startswith("part_"):
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except (IndexError, ValueError):
                continue
    return completed


def main() -> None:
    args = parse_args()

    # Validate input file
    input_path = Path(args.file)
    if not input_path.is_file():
        print(f"{RED}{BOLD}Error: file not found: {args.file}{RESET}")
        sys.exit(1)

    # Check ffmpeg availability
    if not shutil.which("ffmpeg"):
        print(f"{RED}{BOLD}Error: ffmpeg not found on PATH. Install ffmpeg and try again.{RESET}")
        sys.exit(1)

    # Resolve voice file: check as-is first, then in voices/ directory
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.is_file():
            voice_path = Path("voices") / args.voice
        if not voice_path.is_file():
            print(f"{RED}{BOLD}Error: voice file not found: {args.voice}{RESET}")
            print(f"{RED}  Looked in: {args.voice}, voices/{args.voice}{RESET}")
            sys.exit(1)
        args.voice = str(voice_path)

    # Derive book name and create output directory
    book_name = input_path.stem
    book_output_dir = os.path.join(args.output_dir, book_name)
    os.makedirs(book_output_dir, exist_ok=True)

    # Load chunks from disk if they exist, otherwise chunk and save
    txt_files = sorted(Path(book_output_dir).glob("part_*.txt"))
    if txt_files:
        all_indexed_chunks = [
            (int(f.stem.split("_")[1]), f.read_text(encoding="utf-8"))
            for f in txt_files
        ]
        total = len(all_indexed_chunks)
        print(f"{DIM}Loaded {total} chunks from disk.{RESET}")
    else:
        print(f"{DIM}Reading {args.file}...{RESET}")
        text = input_path.read_text(encoding="utf-8")
        print(f"{DIM}File size: {len(text.encode('utf-8')):,} bytes{RESET}")
        raw_chunks = chunk_text(text, args.chunk_size)
        total = len(raw_chunks)
        print(f"{DIM}Text split into {total} chunks, saving to disk...{RESET}")
        for i, chunk in enumerate(raw_chunks, start=1):
            (Path(book_output_dir) / f"part_{i:04d}.txt").write_text(chunk, encoding="utf-8")
        all_indexed_chunks = list(enumerate(raw_chunks, start=1))

    # Apply start-chunk override
    if args.start_chunk:
        all_indexed_chunks = [(i, c) for i, c in all_indexed_chunks if i >= args.start_chunk]
        print(f"{DIM}Starting from chunk {args.start_chunk}.{RESET}")

    # Resume: find already completed chunks
    completed = get_completed_indices(book_output_dir)
    pending = [
        (i, chunk) for i, chunk in all_indexed_chunks if i not in completed
    ]

    if not pending:
        print(f"{GREEN}{BOLD}All chunks already processed. Nothing to do.{RESET}")
        sys.exit(0)

    print(f"{DIM}Resuming: {len(completed)} done, {len(pending)} remaining.{RESET}")

    # Hardware + model
    device = args.device if args.device else get_device()
    print(f"{BLUE}Using device: {device}{RESET}")

    print(f"{YELLOW}Loading OmniVoice model (this may take a while on first run)...{RESET}")
    model = load_model(device)
    print(f"{GREEN}{BOLD}Model loaded.{RESET}")

    prompt = create_prompt(model, args.voice)

    # Generate audio chunk by chunk, saving immediately for resume support
    sample_rate = model.sampling_rate
    total_audio_duration = 0.0
    t_start = time.perf_counter()

    print(f"{BOLD}Starting TTS generation (steps={args.num_steps}, batch={args.tts_batch})...{RESET}")
    batches_done = 0
    total_elapsed = 0.0
    for batch_start in range(0, len(pending), args.tts_batch):
        batch = pending[batch_start : batch_start + args.tts_batch]
        texts = [t for _, t in batch]
        first_idx, last_idx = batch[0][0], batch[-1][0]
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        now = datetime.datetime.now().strftime("%H:%M")
        print(f"{CYAN}  [{first_idx}-{last_idx}/{total}] {now} — Generating {len(batch)} chunks ({total_bytes:,} bytes)...{RESET}")

        t0 = time.perf_counter()
        kwargs = {"text": texts, "language": args.language, "num_step": args.num_steps}
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        if args.instruct:
            kwargs["instruct"] = args.instruct
        audios = model.generate(**kwargs)
        elapsed = time.perf_counter() - t0
        batches_done += 1
        total_elapsed += elapsed

        batch_duration = 0.0
        for (idx, _), audio in zip(batch, audios):
            duration = len(audio) / sample_rate
            total_audio_duration += duration
            batch_duration += duration
            wav_path = os.path.join(book_output_dir, f"part_{idx:04d}.wav")
            save_wav(audio, wav_path, sample_rate)

        rtf = batch_duration / elapsed if elapsed > 0 else 0
        batches_remaining = (len(pending) - batch_start - len(batch)) / args.tts_batch
        avg_batch_time = total_elapsed / batches_done
        eta = datetime.datetime.now() + datetime.timedelta(seconds=batches_remaining * avg_batch_time)
        print(
            f"{CYAN}  [{first_idx}-{last_idx}/{total}]{RESET} "
            f"{batch_duration:.1f}s audio in {elapsed:.1f}s "
            f"(RTF {YELLOW}{rtf:.1f}x{RESET}) — "
            f"ETA {GREEN}{eta.strftime('%H:%M')}{RESET}"
        )

    t_gen = time.perf_counter() - t_start
    print(
        f"{GREEN}{BOLD}TTS complete: {total_audio_duration:.0f}s audio generated "
        f"in {t_gen:.0f}s (RTF {total_audio_duration / t_gen:.1f}x realtime){RESET}"
    )

    # Free model from GPU/MPS memory before CPU-bound MP3 conversion
    del model, prompt
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{DIM}Model freed from memory.{RESET}")

    # Convert WAV → MP3 in parallel
    print(f"{YELLOW}Converting WAV to MP3...{RESET}")
    t_mp3 = time.perf_counter()
    mp3_files = convert_wavs_to_mp3(book_output_dir, workers=args.workers)
    t_mp3 = time.perf_counter() - t_mp3
    print(f"{DIM}Conversion complete: {len(mp3_files)} MP3 files in {t_mp3:.1f}s.{RESET}")

    print(f"{GREEN}{BOLD}Done. Output in {book_output_dir}/{RESET}")


if __name__ == "__main__":
    main()
