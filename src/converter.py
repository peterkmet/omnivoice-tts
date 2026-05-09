from __future__ import annotations

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Write a numpy audio array to a WAV file."""
    sf.write(path, audio, sample_rate)


def _convert_single(wav_path: str) -> str:
    """Convert a single WAV file to MP3 using ffmpeg, then delete the WAV."""
    mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", mp3_path],
        check=True,
        capture_output=True,
    )
    os.remove(wav_path)
    return mp3_path


def convert_wavs_to_mp3(wav_dir: str, workers: int) -> list[str]:
    """Convert all WAV files in a directory to MP3 in parallel.

    Returns list of created MP3 file paths.
    """
    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    if not wav_files:
        return []

    wav_paths = [str(f) for f in wav_files]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        mp3_paths = list(pool.map(_convert_single, wav_paths))

    return mp3_paths
