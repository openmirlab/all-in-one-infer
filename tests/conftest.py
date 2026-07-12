"""Shared fixtures for the end-to-end tests (test_analyze / test_sonify /
test_visualize).

The session-scoped `analysis_result` fixture runs `allin1_infer.analyze()`
ONCE on the bundled demo track (assets/Sunflower 60BPM.mp3) and shares the
result across all three E2E test modules, so the heavyweight part (demucs
source separation + 8-checkpoint harmonix-all ensemble inference) is paid a
single time per pytest session. Model weights are auto-downloaded to the HF /
demucs caches on first use, which needs network; the demo track ships in the
repo (not in the wheel), so these tests only run from a source checkout.

Byproducts (demix stems, spectrograms) go to a pytest-managed session tmp dir
and are never written into the repo.
"""

import os
from pathlib import Path

import pytest

# The E2E plots must never try to open a GUI window under pytest.
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO_ROOT = Path(__file__).resolve().parent.parent
# A .wav demo track, deliberately: sonify's save path mirrors the source
# suffix ({stem}.sonif{suffix}), and writing .mp3 would require the optional
# demucs-infer[mp3] lameenc extra -- .wav keeps the whole E2E suite runnable
# on a plain core install.
TEST_TRACK = REPO_ROOT / 'assets' / "NewJeans (뉴진스) 'Super Shy' Official MV.wav"


@pytest.fixture(scope='session')
def analysis_result(tmp_path_factory):
  """One full analyze() run on the bundled demo track, shared session-wide."""
  if not TEST_TRACK.is_file():
    pytest.skip(
      f'demo track not found: {TEST_TRACK} '
      '(E2E tests need the repo assets/, e.g. running from an sdist/wheel)'
    )

  import allin1_infer

  work = tmp_path_factory.mktemp('allin1-e2e')
  return allin1_infer.analyze(
    TEST_TRACK,
    demix_dir=work / 'demix',
    spec_dir=work / 'spec',
  )
