# OmniVoice Book Converter

Local CLI tool that converts `.txt` books into MP3 audiobooks using [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS. Runs entirely offline on Apple Silicon (MPS) and NVIDIA (CUDA). No data leaves the device.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) on `$PATH`

## Setup

```bash
git clone https://github.com/peterkmet/omnivoice-tts.git
cd omnivoice-tts
uv sync
```

Model weights (~3.3 GB) are downloaded from HuggingFace on first run and cached locally.

## Usage

```bash
uv run python src/main.py --file convert/book.txt --language en --workers 1
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--file` | yes | — | Path to input `.txt` file |
| `--language` | no | `en` | Language: `en`, `sk`, `cs` |
| `--voice` | no | model default | Voice reference sample (filename in `voices/` or full path) |
| `--instruct` | no | — | Voice design attributes, e.g. `"female, young adult, high pitch"` |
| `--output-dir` | no | `./output` | Output directory |
| `--chunk-size` | no | `10240` | Chunk size in bytes |
| `--workers` | no | CPU count | Parallel ffmpeg workers for MP3 conversion |
| `--tts-batch` | no | `4` | Number of chunks per GPU batch |
| `--num-steps` | no | `16` | Diffusion steps per chunk (max quality: `32`) |
| `--device` | no | auto | Force device: `mps`, `cuda`, or `cpu` |
| `--start-chunk` | no | — | Resume from this chunk number (1-based) |

### Examples

Basic conversion:
```bash
uv run python src/main.py --file convert/book.txt --language cs --workers 1
```

With voice cloning (3–10s reference clip):
```bash
uv run python src/main.py --file convert/book.txt --language cs --voice narrator.wav
```

With voice design:
```bash
uv run python src/main.py --file convert/book.txt --language en \
  --instruct "male, middle-aged, low pitch, british accent"
```

### Voice Design Attributes

Combine attributes with commas. One per category:

- **Gender:** `male`, `female`
- **Age:** `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- **Pitch:** `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`
- **Style:** `whisper`
- **Accents (English only):** `american accent`, `british accent`, `australian accent`, `indian accent`, ...

Voice design was trained on English and Chinese. For other languages, voice cloning (`--voice`) is more reliable.

## Output

```
output/
  book/
    part_0001.mp3
    part_0002.mp3
    ...
```

Text chunks are saved as `part_XXXX.txt` alongside the audio files. On re-run, chunks are loaded from disk and already completed audio is skipped automatically.

To resume from a specific chunk:
```bash
uv run python src/main.py --file convert/book.txt --language cs --start-chunk 42
```

## Tests

```bash
uv run pytest                                              # unit tests
uv run pytest tests/test_inference_integration.py -v -s    # integration test (loads model)
```

## Supported Languages

- English (`en`)
- Slovak (`sk`)
- Czech (`cs`)

## License

MIT
