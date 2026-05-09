from __future__ import annotations

import time

import numpy as np
import torch
from omnivoice import OmniVoice


def load_model(device: str) -> OmniVoice:
    """Load OmniVoice model onto the specified device."""
    return OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
    )


def create_prompt(model: OmniVoice, voice_path: str | None):
    """Create a reusable voice clone prompt from a reference audio file."""
    if voice_path:
        return model.create_voice_clone_prompt(ref_audio=voice_path)
    return None


def generate_chunks(
    model: OmniVoice,
    chunks: list[str],
    language: str,
    prompt=None,
    batch_size: int = 1,
) -> list[np.ndarray]:
    """Generate audio for text chunks in batches.

    Returns a list of numpy arrays (one per chunk) at model.sampling_rate Hz.
    """
    results = []
    total = len(chunks)
    sample_rate = model.sampling_rate

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_end = min(i + batch_size, total)
        batch_bytes = sum(len(t.encode("utf-8")) for t in batch)
        print(f"  Generating chunks {i + 1}-{batch_end}/{total} ({batch_bytes:,} bytes)...")

        t0 = time.perf_counter()
        kwargs = {"text": batch, "language": language}
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt
        audios = model.generate(**kwargs)
        elapsed = time.perf_counter() - t0

        batch_duration = sum(len(a) / sample_rate for a in audios)
        rtf = batch_duration / elapsed if elapsed > 0 else 0
        print(
            f"    Done in {elapsed:.1f}s → {batch_duration:.1f}s audio "
            f"(RTF {rtf:.1f}x realtime)"
        )

        results.extend(audios)

    return results
