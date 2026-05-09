import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from converter import save_wav, _convert_single, convert_wavs_to_mp3


@pytest.fixture
def sample_audio():
    """Generate 1 second of 440 Hz sine wave at 24 kHz."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def wav_file(tmp_path, sample_audio):
    """Save sample audio as a WAV file and return the path."""
    audio, sr = sample_audio
    path = str(tmp_path / "part_0001.wav")
    save_wav(audio, path, sr)
    return path


class TestSaveWav:
    def test_creates_file(self, tmp_path, sample_audio):
        audio, sr = sample_audio
        path = str(tmp_path / "test.wav")
        save_wav(audio, path, sr)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_correct_sample_rate(self, tmp_path, sample_audio):
        import soundfile as sf

        audio, sr = sample_audio
        path = str(tmp_path / "test.wav")
        save_wav(audio, path, sr)
        info = sf.info(path)
        assert info.samplerate == sr

    def test_correct_duration(self, tmp_path, sample_audio):
        import soundfile as sf

        audio, sr = sample_audio
        path = str(tmp_path / "test.wav")
        save_wav(audio, path, sr)
        info = sf.info(path)
        assert abs(info.duration - 1.0) < 0.01


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
class TestConvertSingle:
    def test_creates_mp3(self, wav_file):
        mp3_path = _convert_single(wav_file)
        assert Path(mp3_path).exists()
        assert mp3_path.endswith(".mp3")

    def test_removes_wav(self, wav_file):
        _convert_single(wav_file)
        assert not Path(wav_file).exists()

    def test_mp3_not_empty(self, wav_file):
        mp3_path = _convert_single(wav_file)
        assert Path(mp3_path).stat().st_size > 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
class TestConvertWavsToMp3:
    def test_batch_conversion(self, tmp_path, sample_audio):
        audio, sr = sample_audio
        # Create 3 WAV files
        for i in range(1, 4):
            save_wav(audio, str(tmp_path / f"part_{i:04d}.wav"), sr)

        mp3_paths = convert_wavs_to_mp3(str(tmp_path), workers=2)
        assert len(mp3_paths) == 3
        for mp3 in mp3_paths:
            assert Path(mp3).exists()

        # All WAVs should be deleted
        assert list(tmp_path.glob("*.wav")) == []

    def test_empty_directory(self, tmp_path):
        result = convert_wavs_to_mp3(str(tmp_path), workers=2)
        assert result == []

    def test_preserves_naming(self, tmp_path, sample_audio):
        audio, sr = sample_audio
        save_wav(audio, str(tmp_path / "part_0042.wav"), sr)

        mp3_paths = convert_wavs_to_mp3(str(tmp_path), workers=1)
        assert len(mp3_paths) == 1
        assert Path(mp3_paths[0]).name == "part_0042.mp3"
