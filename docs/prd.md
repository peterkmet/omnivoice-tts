# PRD: OmniVoice Book Converter

## 1. Project Definition
Local Python-based application for high-quality conversion of long text documents (books) into speech (TTS) utilizing the OmniVoice framework. The system is optimized for local execution on Apple Silicon (M3 Max) and NVIDIA GPUs, ensuring total data privacy.

## 2. Personas

### B2C (Consumer)
* **Audiobook Enthusiast:** Users with large `.txt` files who want to listen to titles during transit or exercise with professional-grade voice quality.
* **Student/Academic:** Individuals needing to convert study materials and scripts into audio for passive learning and revision.

### B2B (Business)
* **Independent Authors & Small Publishers:** Cost-effective solution for creating demo tracks or final audiobooks without studio overhead.
* **Accessibility Developers:** For integrating local, privacy-compliant TTS into specialized hardware or closed systems for the visually impaired.

## 3. Main User Flow
1. **Source Input:** User places a `.txt` file into the `/convert` directory.
2. **Validation & Parsing:** User runs the script. The application reads the file, verifies encoding, and segments the text into ~10KB chunks ensuring no sentence is split mid-way (split on `.`, `?`, `!`).
3. **Inference Configuration:** System identifies available hardware (MPS/CUDA) and loads the OmniVoice model. Optionally loads a custom voice sample from `/voices/`.
4. **Audio Generation:** Parallel TTS inference of chunks into intermediate `.wav` files, utilizing all available cores on M3 Max.
5. **MP3 Conversion:** After all chunks are generated, a parallel ffmpeg batch converts `.wav` files to `.mp3` (CPU-bound, independent from GPU inference).
6. **Export:** Final `.mp3` files are saved in a structured subdirectory within `/output` named after the source file (e.g., `/output/my_book/part_0001.mp3`). Intermediate `.wav` files are removed.
7. **Resume:** If re-run, the script detects existing output and resumes from the first missing chunk.

## 4. User Stories (Gherkin)

### Story 1: Automatic Text Segmentation
* **Given** a `.txt` file in the `/convert` directory.
* **When** the conversion process is initiated.
* **Then** the system splits the text into chunks of approximately 10KB, ensuring the cut always occurs at a sentence boundary (period, question mark, or exclamation point).

### Story 2: Hardware Acceleration on M3 Max
* **Given** the application is running on an Apple Silicon M3 Max device.
* **When** the TTS inference process starts.
* **Then** the application prioritizes `torch.backends.mps` over the CPU to maximize generation speed.

### Story 3: Parallel Processing
* **Given** a text file has been split into N chunks.
* **When** the TTS inference runs.
* **Then** the system processes chunks in parallel, utilizing all available M3 Max cores to maximize throughput.

### Story 4: Custom Voice Sample
* **Given** a `.wav` or `.mp3` file exists in the `/voices/` directory.
* **When** the user specifies the voice sample (or a default is used).
* **Then** the OmniVoice model uses that sample as reference for voice cloning.

### Story 5: Multi-language Support
* **Given** a text file in English, Slovak, or Czech.
* **When** the conversion process is initiated.
* **Then** the system generates speech in the corresponding language.

### Story 6: Resumable Processing (State Persistence)
* **Given** a conversion was interrupted (crash, user abort, etc.).
* **When** the user re-runs the script for the same file.
* **Then** the system checks `/output/<book_name>/` for already generated chunks and resumes from the first missing chunk, skipping completed ones.

### Story 7: CLI Interface
* **Given** the user runs the script from the command line.
* **When** providing arguments such as `--file`, `--voice`, `--language`, `--output-dir`.
* **Then** the system uses the provided values to configure the conversion. All arguments except `--file` have sensible defaults (language: `en`, chunk size: `10240` bytes, workers: number of CPU cores).

## 5. MVP Checklist & Acceptance Criteria
- [ ] **Project Setup**
    - *AC:* OmniVoice and all dependencies installed into `.venv` via `uv`. Setup documented.
- [ ] **Text Chunking Module**
    - *AC:* Text is split into ~10KB chunks. No chunk splits mid-sentence. Splits occur at sentence boundaries (`.`, `?`, `!`).
- [ ] **OmniVoice Integration**
    - *AC:* Successful loading of local model weights without external API calls.
- [ ] **MPS and CUDA Support**
    - *AC:* Automatic detection and backend switching based on the operating system without manual code changes.
- [ ] **Parallel Processing**
    - *AC:* Chunks are processed in parallel, utilizing all available cores on M3 Max.
- [ ] **Voice Sample Support**
    - *AC:* Default voice works out of the box. Custom voice sample from `/voices/` directory is used when provided.
- [ ] **Multi-language Support**
    - *AC:* English, Slovak, and Czech text is correctly synthesized.
- [ ] **MP3 Output**
    - *AC:* TTS inference generates intermediate `.wav` files. After all chunks are complete, a parallel ffmpeg batch converts them to `.mp3`. The ffmpeg conversion uses CPU parallelism independently from model inference.
- [ ] **File Management**
    - *AC:* Each book has its own folder with sequentially indexed files.
- [ ] **Resumable Processing**
    - *AC:* On re-run, the application detects existing chunks in `/output/<book_name>/` and skips already generated ones. Only missing chunks are processed.
- [ ] **CLI Interface**
    - *AC:* Supported arguments: `--file` (required), `--voice` (optional, path to voice sample), `--language` (optional, default: `en`), `--output-dir` (optional, default: `./output`), `--chunk-size` (optional, default: `10240` bytes), `--workers` (optional, default: number of CPU cores). `--help` prints usage.

## 6. Tech Stack & Non-negotiables

### Tech Stack
* **Runtime:** Python 3.10+
* **Package Manager:** uv
* **AI Framework:** PyTorch (with MPS for macOS and CUDA for Windows/Linux support).
* **TTS Engine:** k2-fsa/OmniVoice.
* **Processing:** NumPy, Re (Regex).
* **Audio Encoding:** ffmpeg (for MP3 conversion).

### Non-negotiables
* **Zero Cloud:** No data shall leave the local device (Privacy by Design).
* **English Documentation:** All code documentation and comments must be in English.
* **No Unsolicited Code:** Code generation requires an explicit prompt.

## 7. Out of Scope
* **GUI (Graphical User Interface):** Initial phase is strictly CLI/Script-based.
* **Audio Post-production:** Noise removal, volume normalization, or merging files into `.m4b`.
* **EPUB Support:** Only `.txt` input is supported in MVP.
* **OCR:** Converting PDFs or images to text is not part of the MVP.
* **Mobile Platforms:** Desktop OS only (macOS, Windows, Linux).
* **Chapter Detection:** MVP uses fixed-size chunking (~10KB), not semantic chapter splitting.

## 8. Success Metrics
1. **Inference Speed:** Achieve a generation ratio of at least 1:5 (1 minute of audio generated in 12 seconds on M3 Max).
2. **Stability:** 100% success rate in processing `.txt` files up to 5MB without application crashes.
3. **Portability:** Identical code successfully running on both macOS (MPS) and Windows (CUDA) without logic modification.
4. **Parallelism:** Full utilization of available compute cores during inference.
