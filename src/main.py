from __future__ import annotations

import argparse
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
        default=16,
        help="Diffusion steps per chunk (default: 16, max quality: 32)",
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
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    # Check ffmpeg availability
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found on PATH. Install ffmpeg and try again.")
        sys.exit(1)

    # Resolve voice file: check as-is first, then in voices/ directory
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.is_file():
            voice_path = Path("voices") / args.voice
        if not voice_path.is_file():
            print(f"Error: voice file not found: {args.voice}")
            print(f"  Looked in: {args.voice}, voices/{args.voice}")
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
        print(f"Loaded {total} chunks from disk.")
    else:
        print(f"Reading {args.file}...")
        text = input_path.read_text(encoding="utf-8")
        print(f"File size: {len(text.encode('utf-8')):,} bytes")
        raw_chunks = chunk_text(text, args.chunk_size)
        total = len(raw_chunks)
        print(f"Text split into {total} chunks, saving to disk...")
        for i, chunk in enumerate(raw_chunks, start=1):
            (Path(book_output_dir) / f"part_{i:04d}.txt").write_text(chunk, encoding="utf-8")
        all_indexed_chunks = list(enumerate(raw_chunks, start=1))

    # Apply start-chunk override
    if args.start_chunk:
        all_indexed_chunks = [(i, c) for i, c in all_indexed_chunks if i >= args.start_chunk]
        print(f"Starting from chunk {args.start_chunk}.")

    # Resume: find already completed chunks
    completed = get_completed_indices(book_output_dir)
    pending = [
        (i, chunk) for i, chunk in all_indexed_chunks if i not in completed
    ]

    if not pending:
        print("All chunks already processed. Nothing to do.")
        sys.exit(0)

    print(f"Resuming: {len(completed)} done, {len(pending)} remaining.")

    # Hardware + model
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    print("Loading OmniVoice model (this may take a while on first run)...")
    model = load_model(device)
    print("Model loaded.")

    prompt = create_prompt(model, args.voice)

    # Generate audio chunk by chunk, saving immediately for resume support
    sample_rate = model.sampling_rate
    total_audio_duration = 0.0
    t_start = time.perf_counter()

    print(f"Starting TTS generation (steps={args.num_steps}, batch={args.tts_batch})...")
    for batch_start in range(0, len(pending), args.tts_batch):
        batch = pending[batch_start : batch_start + args.tts_batch]
        texts = [t for _, t in batch]
        first_idx, last_idx = batch[0][0], batch[-1][0]
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        print(f"  [{first_idx}-{last_idx}/{total}] Generating {len(batch)} chunks ({total_bytes:,} bytes)...")

        t0 = time.perf_counter()
        kwargs = {"text": texts, "language": args.language, "num_step": args.num_steps}
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        if args.instruct:
            kwargs["instruct"] = args.instruct
        audios = model.generate(**kwargs)
        elapsed = time.perf_counter() - t0

        batch_duration = 0.0
        for (idx, _), audio in zip(batch, audios):
            duration = len(audio) / sample_rate
            total_audio_duration += duration
            batch_duration += duration
            wav_path = os.path.join(book_output_dir, f"part_{idx:04d}.wav")
            save_wav(audio, wav_path, sample_rate)

        rtf = batch_duration / elapsed if elapsed > 0 else 0
        print(f"  [{first_idx}-{last_idx}/{total}] {batch_duration:.1f}s audio in {elapsed:.1f}s (RTF {rtf:.1f}x)")

    t_gen = time.perf_counter() - t_start
    print(
        f"TTS complete: {total_audio_duration:.0f}s audio generated "
        f"in {t_gen:.0f}s (RTF {total_audio_duration / t_gen:.1f}x realtime)"
    )

    # Free model from GPU/MPS memory before CPU-bound MP3 conversion
    del model, prompt
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model freed from memory.")

    # Convert WAV → MP3 in parallel
    print("Converting WAV to MP3...")
    t_mp3 = time.perf_counter()
    mp3_files = convert_wavs_to_mp3(book_output_dir, workers=args.workers)
    t_mp3 = time.perf_counter() - t_mp3
    print(f"Conversion complete: {len(mp3_files)} MP3 files in {t_mp3:.1f}s.")

    print(f"Done. Output in {book_output_dir}/")


if __name__ == "__main__":
    main()
