"""TorchAudio-free import and input-audio loading contracts (issue #7).

`stems.py` used to `import torchaudio` at module top level while the base
install never declared it -- it only arrived transitively via demucs-infer,
which dropped it at ffe0080, so a fresh install failed on `import
allin1_infer`. These tests pin the replacement: no torchaudio import or
metadata requirement, wav/flac via soundfile, every other format via
demucs-infer's ffmpeg-backed `AudioFile` with no silent soundfile fallback.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from allin1_infer import stems

if sys.version_info >= (3, 11):
  import tomllib
else:
  import tomli as tomllib

_ROOT = Path(__file__).resolve().parent.parent
_needs_ffmpeg = pytest.mark.skipif(
  shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
  reason='ffmpeg/ffprobe executables not on PATH',
)


def test_package_imports_when_torchaudio_is_unavailable():
  code = """
import importlib.abc
import sys

class BlockTorchAudio(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == 'torchaudio' or fullname.startswith('torchaudio.'):
      raise RuntimeError('torchaudio imported')

sys.meta_path.insert(0, BlockTorchAudio())
import allin1_infer
from allin1_infer import stems
assert callable(allin1_infer.analyze) and callable(stems._load_input_audio)
"""
  completed = subprocess.run(
    [sys.executable, '-c', code], cwd=_ROOT, capture_output=True, text=True, check=False,
  )
  assert completed.returncode == 0, completed.stderr


def test_package_metadata_does_not_pull_torchaudio_or_torchcodec():
  with (_ROOT / 'pyproject.toml').open('rb') as stream:
    project = tomllib.load(stream)['project']
  requirements = list(project['dependencies'])
  for extra in project.get('optional-dependencies', {}).values():
    requirements.extend(extra)
  assert all(not requirement.lower().startswith(('torchaudio', 'torchcodec'))
             for requirement in requirements)


def _fail_audio_file(path):
  raise AssertionError('wav/flac must not go through ffmpeg')


@pytest.mark.parametrize('suffix', ['.wav', '.flac'])
def test_lossless_input_is_read_via_soundfile(tmp_path, monkeypatch, suffix):
  monkeypatch.setattr(stems, 'AudioFile', _fail_audio_file)
  data = np.random.default_rng(0).uniform(-0.5, 0.5, size=(2205, 2)).astype(np.float32)
  path = tmp_path / f'input{suffix}'
  sf.write(str(path), data, 22050, subtype='PCM_24')
  wav, sr = stems._load_input_audio(path)
  assert sr == 22050
  assert wav.dtype == torch.float32 and wav.shape == (2, 2205)
  expected, _ = sf.read(str(path), dtype='float32', always_2d=True)
  assert np.array_equal(wav.numpy(), expected.T)


@pytest.mark.parametrize('error, fragment', [
  (FileNotFoundError(2, 'No such file or directory', 'ffprobe'), 'ffmpeg/ffprobe executables were not found'),
  (subprocess.CalledProcessError(1, ['ffmpeg']), 'ffmpeg could not decode it'),
])
def test_lossy_input_failure_is_actionable_and_never_falls_back(tmp_path, monkeypatch, error, fragment):
  def raising_audio_file(path):
    raise error

  def fail_soundfile(*args, **kwargs):
    raise AssertionError('lossy input must not fall back to soundfile')

  monkeypatch.setattr(stems, 'AudioFile', raising_audio_file)
  monkeypatch.setattr(stems.sf, 'read', fail_soundfile)
  with pytest.raises(RuntimeError) as excinfo:
    stems._load_input_audio(tmp_path / 'input.mp3')
  message = str(excinfo.value)
  assert fragment in message
  assert 'convert the file to wav/flac' in message
  assert 'torchaudio' not in message and 'torchcodec' not in message


def _encode_mp3(tmp_path, channels, samplerate):
  source = tmp_path / 'source.wav'
  t = np.arange(samplerate // 2) / samplerate
  tone = 0.3 * np.sin(2 * np.pi * 440 * t)
  sf.write(str(source), np.stack([tone] * channels, axis=1), samplerate)
  target = tmp_path / 'input.mp3'
  subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(source), str(target)], check=True)
  return target


@_needs_ffmpeg
@pytest.mark.parametrize('channels, samplerate', [(1, 48000), (2, 44100)])
def test_mp3_input_keeps_native_rate_and_channels(tmp_path, channels, samplerate):
  wav, sr = stems._load_input_audio(_encode_mp3(tmp_path, channels, samplerate))
  assert sr == samplerate
  assert wav.dtype == torch.float32 and wav.shape[0] == channels
  assert wav.shape[1] >= samplerate // 2 and wav.abs().max() > 0.1


@_needs_ffmpeg
def test_mp3_decode_matches_torchaudio_ffmpeg_backend_when_available(tmp_path):
  """Parity with the decoder this replaced; skipped where torchaudio is absent."""
  torchaudio = pytest.importorskip('torchaudio')
  path = _encode_mp3(tmp_path, 2, 44100)
  try:
    expected, expected_sr = torchaudio.load(str(path))
  except Exception as err:  # torchaudio>=2.11 without torchcodec cannot decode
    pytest.skip(f'torchaudio cannot decode mp3 here: {err}')
  wav, sr = stems._load_input_audio(path)
  assert sr == expected_sr
  assert torch.equal(wav, expected)
