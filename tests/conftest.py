"""Offline-by-default tests, with explicit model-backed integration opt-in.

Integration tests share one full CPU analysis of caller-provided WAV audio. They may
populate model caches; all inference byproducts stay in pytest-managed temporary paths.
Reads: pytest options and markers · allin1_infer.analyze (only after explicit opt-in).
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')


def pytest_addoption(parser):
  parser.addoption(
    '--run-integration', action='store_true', default=False,
    help='Run model-backed tests; permits checkpoint downloads and requires --integration-audio.',
  )
  parser.addoption(
    '--integration-audio', metavar='WAV',
    help='Existing, caller-provided music WAV used by the model-backed integration tests.',
  )


def pytest_configure(config):
  config.addinivalue_line('markers', 'integration: full model inference on external audio')
  config.addinivalue_line('markers', 'network: may download model checkpoints on first use')
  audio = config.getoption('--integration-audio')
  enabled = config.getoption('--run-integration')
  if audio and not enabled:
    raise pytest.UsageError('--integration-audio requires --run-integration')
  if enabled:
    if not audio:
      raise pytest.UsageError('--run-integration requires --integration-audio /path/to/music.wav')
    path = Path(audio).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != '.wav':
      raise pytest.UsageError(f'--integration-audio must be an existing WAV file: {path}')


def pytest_collection_modifyitems(config, items):
  if config.getoption('--run-integration'):
    return
  selected, deselected = [], []
  for item in items:
    (deselected if item.get_closest_marker('integration') else selected).append(item)
  items[:] = selected
  config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope='session')
def analysis_result(tmp_path_factory, pytestconfig):
  """Run one explicitly requested full-model analysis; missing input is an error."""
  import allin1_infer

  audio = Path(pytestconfig.getoption('--integration-audio')).expanduser().resolve()
  work = tmp_path_factory.mktemp('allin1-e2e')
  return allin1_infer.analyze(
    audio,
    device='cpu',
    multiprocess=False,
    demix_dir=work / 'demix',
    spec_dir=work / 'spec',
  )
