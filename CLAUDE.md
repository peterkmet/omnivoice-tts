# CLAUDE.md — OmniVoice Book Converter

## Project Overview
Local CLI application that converts `.txt` books into MP3 audiobooks using **k2-fsa/OmniVoice** TTS. Runs entirely offline on Apple Silicon (MPS) and NVIDIA (CUDA). No data leaves the device.

## Architecture

```
convert/          <- input .txt files
voices/           <- optional custom voice samples (.wav/.mp3)
output/
  <book_name>/
    part_0001.wav <- intermediate (deleted after MP3 conversion)
    part_0001.mp3 <- final output
src/
  main.py         <- CLI entrypoint
  chunker.py      <- text segmentation (~10KB, sentence boundary)
  inference.py    <- OmniVoice model loading & TTS generation
  hardware.py     <- MPS/CUDA/CPU detection
  converter.py    <- ffmpeg wav->mp3 batch conversion
docs/
  prd.md          <- full product requirements
```

## Tech Stack
- **Python 3.10+** with `uv` (package manager, virtual env in `.venv`)
- **PyTorch** — MPS (macOS) / CUDA (Windows/Linux) / CPU fallback
- **k2-fsa/OmniVoice** — TTS engine, local weights only
- **ffmpeg** — WAV to MP3 conversion (must be installed on system)
- **NumPy**, **re** — text processing

## CLI Usage
```bash
python src/main.py --file convert/book.txt \
  --language en \
  --voice voices/sample.wav \
  --output-dir ./output \
  --chunk-size 10240 \
  --workers 8
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--file` | yes | — | Path to input `.txt` file |
| `--language` | no | `en` | Language: `en`, `sk`, `cs` |
| `--voice` | no | model default | Path to voice reference sample |
| `--output-dir` | no | `./output` | Output directory |
| `--chunk-size` | no | `10240` | Chunk size in bytes |
| `--workers` | no | CPU count | Number of parallel workers |

## Key Design Decisions

### Text Chunking
- Split text into chunks of `--chunk-size` bytes (default 10KB).
- Never split mid-sentence. Cut at sentence boundaries (`.`, `?`, `!`).
- Chunks are indexed sequentially: `part_0001`, `part_0002`, etc.

### Processing Pipeline
1. Read and validate `.txt` file (UTF-8).
2. Chunk text into segments.
3. Parallel TTS inference → intermediate `.wav` files.
4. Parallel ffmpeg batch → `.mp3` files (CPU-only, independent from GPU inference).
5. Remove intermediate `.wav` files.

### Resumable Processing
- On re-run, check `/output/<book_name>/` for existing `.wav`/`.mp3` files.
- Skip already completed chunks, resume from first missing one.

### Hardware Detection
- Auto-detect: MPS (macOS) > CUDA (Linux/Windows) > CPU.
- No manual code changes needed to switch backend.

## Coding Conventions
- All code, comments, and documentation in **English**.
- No external API calls. Zero cloud. Privacy by design.
- No unsolicited code — every change requires an explicit prompt.
- Keep it simple. No over-engineering. MVP first.

## Workflow Rules
- **Never generate code without approval.** Always describe what you plan to change and wait for explicit confirmation before writing any code.

## Languages Supported
- English (`en`)
- Slovak (`sk`)
- Czech (`cs`)

## Out of Scope (MVP)
- GUI — CLI only
- EPUB / PDF / OCR input
- Audio post-production (normalization, noise removal, m4b merging)
- Chapter detection (fixed-size chunking only)
- Mobile platforms

## Running Tests
```bash
uv run pytest
```

## Session Start
At the beginning of every session, always read the documentation in `docs/` to refresh context on current requirements and project state.

## Compact Instructions
When compacting context, preserve in this order:
1. Architectural decisions (NEVER summarize)
2. Modified files and key changes
3. Verification state (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can be deleted, keep only pass/fail)

## Setup
```bash
uv venv
uv pip install -r requirements.txt
# or: uv sync (if using pyproject.toml)
```
OmniVoice model weights are installed locally into `.venv` as part of the dependency setup. ffmpeg must be available on `$PATH`.
