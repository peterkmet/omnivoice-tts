"""Integration test: generate audio from tests/sample.txt and save output.

Run with:
    uv run pytest tests/test_inference_integration.py -v -s

Uses -s to show print output (timing, speed, audio stats).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hardware import get_device
from inference import load_model, generate_chunks

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = TESTS_DIR / "sample.txt"
OUTPUT_WAV = TESTS_DIR / "sample_output.wav"


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    device = get_device()
    print(f"\nLoading model on {device}...")
    t0 = time.perf_counter()
    m = load_model(device)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")
    return m


class TestSingleChunkInference:
    LANGUAGE = "cs"

    def test_generates_audio(self, model):
        text = SAMPLE_FILE.read_text(encoding="utf-8").strip()
        text_bytes = len(text.encode("utf-8"))
        print(f"\n--- Single chunk inference test ---")
        print(f"Input: {SAMPLE_FILE}")
        print(f"Text: {text!r}")
        print(f"Text size: {text_bytes} bytes")
        print(f"Language: {self.LANGUAGE}")

        t0 = time.perf_counter()
        audios = generate_chunks(
            model,
            [text],
            language=self.LANGUAGE,
            batch_size=1,
        )
        elapsed = time.perf_counter() - t0

        assert len(audios) == 1
        audio = audios[0]

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) > 0

        sample_rate = model.sampling_rate
        duration = len(audio) / sample_rate
        rtf = duration / elapsed if elapsed > 0 else 0

        # Save output
        sf.write(str(OUTPUT_WAV), audio, sample_rate)

        print(f"\n--- Results ---")
        print(f"Output: {OUTPUT_WAV}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Samples: {len(audio):,}")
        print(f"Audio duration: {duration:.2f}s")
        print(f"Inference time: {elapsed:.2f}s")
        print(f"RTF: {rtf:.1f}x realtime")
        print(f"Audio min/max: {audio.min():.4f} / {audio.max():.4f}")
        print(f"Audio RMS: {np.sqrt(np.mean(audio ** 2)):.4f}")
        print(f"File size: {OUTPUT_WAV.stat().st_size:,} bytes")

        # Sanity checks
        assert sample_rate == 24000
        assert duration > 0.5, "Audio too short for 3 sentences"
        assert duration < 30.0, "Audio unreasonably long"
        assert np.abs(audio).max() > 0.01, "Audio is silence"
        assert OUTPUT_WAV.exists(), "Output WAV not saved"
