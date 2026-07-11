"""Tests for AnalysisResult.activation_fps (issue #2): the frame rate of the
`activations` arrays (model's `cfg.fps`) must ride along with the result, and
must round-trip through save_results/from_json -- including old JSONs saved
before this field existed.

These tests build a fake model (real Config, fake forward pass) rather than
downloading a pretrained checkpoint, so they run without model weights or a
real audio file.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from allin1_infer.config import Config, HarmonixConfig
from allin1_infer.helpers import run_inference, save_results
from allin1_infer.typings import AllInOneOutput, AnalysisResult, Segment

NUM_LABELS = 10
NUM_FRAMES = 200  # 2s at fps=100


class _FakeModel(torch.nn.Module):
  """Stands in for a real AllInOne/Ensemble model: same `.cfg` + callable
  contract run_inference relies on, but forward() returns fixed logits
  instead of doing a real forward pass."""

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg

  def forward(self, spec):
    torch.manual_seed(0)
    return AllInOneOutput(
      logits_beat=torch.randn(1, NUM_FRAMES),
      logits_downbeat=torch.randn(1, NUM_FRAMES),
      logits_section=torch.randn(1, NUM_FRAMES),
      logits_function=torch.randn(1, NUM_LABELS, NUM_FRAMES),
      embeddings=torch.randn(1, NUM_FRAMES, 8),
    )


def _make_cfg(fps: int = 100) -> Config:
  return Config(data=HarmonixConfig(), fps=fps, best_threshold_downbeat=0.5)


def _run(tmp_path: Path, include_activations: bool, fps: int = 100):
  spec_path = tmp_path / 'spec.npy'
  np.save(spec_path, np.zeros((12, NUM_FRAMES), dtype=np.float32))

  model = _FakeModel(_make_cfg(fps))
  return run_inference(
    path=tmp_path / 'track.mp3',
    spec_path=spec_path,
    model=model,
    device='cpu',
    include_activations=include_activations,
    include_embeddings=False,
  )


def test_run_inference_populates_activation_fps_when_activations_included():
  with tempfile.TemporaryDirectory() as d:
    result = _run(Path(d), include_activations=True, fps=100)

  assert result.activations is not None
  assert result.activation_fps == 100.0


def test_run_inference_leaves_activation_fps_none_without_activations():
  with tempfile.TemporaryDirectory() as d:
    result = _run(Path(d), include_activations=False)

  assert result.activations is None
  assert result.activation_fps is None


def test_run_inference_uses_the_models_own_fps():
  # cfg.fps is embedded per-checkpoint; a model trained/exported with a
  # different fps must report its own value, not a hardcoded 100.
  with tempfile.TemporaryDirectory() as d:
    result = _run(Path(d), include_activations=True, fps=50)

  assert result.activation_fps == 50.0


def test_save_load_roundtrip_preserves_activation_fps():
  result = AnalysisResult(
    path=Path('/tmp/some_track.mp3'),
    bpm=120,
    beats=[0.1, 0.2],
    downbeats=[0.1],
    beat_positions=[1, 2],
    segments=[Segment(0.0, 1.0, 'intro')],
    activations={'beat': np.zeros(3, dtype=np.float32)},
    activation_fps=100.0,
  )

  with tempfile.TemporaryDirectory() as d:
    save_results(result, d)
    json_path = Path(d) / 'some_track.json'

    saved = json.loads(json_path.read_text())
    assert saved['activation_fps'] == 100.0

    loaded = AnalysisResult.from_json(json_path)
    assert loaded.activation_fps == 100.0
    assert loaded.activations is not None
    assert list(loaded.activations['beat']) == [0.0, 0.0, 0.0]


def test_load_old_json_without_activation_fps_defaults_to_none():
  # Simulates a result saved by a pre-fix version of this package: no
  # `activation_fps` key in the JSON at all.
  result = AnalysisResult(
    path=Path('/tmp/legacy_track.mp3'),
    bpm=120,
    beats=[0.1, 0.2],
    downbeats=[0.1],
    beat_positions=[1, 2],
    segments=[Segment(0.0, 1.0, 'intro')],
  )

  with tempfile.TemporaryDirectory() as d:
    save_results(result, d)
    json_path = Path(d) / 'legacy_track.json'

    data = json.loads(json_path.read_text())
    # save_results wrote the (unset) field as `"activation_fps": null`; drop
    # the key entirely to simulate a genuinely pre-fix JSON that never had it.
    del data['activation_fps']
    json_path.write_text(json.dumps(data))

    loaded = AnalysisResult.from_json(json_path)
    assert loaded.activation_fps is None
