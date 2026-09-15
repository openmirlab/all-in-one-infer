"""Compatibility tests for the metrical DBN decoder construction seam."""

import numpy as np
import torch

from allin1_infer.config import Config, HarmonixConfig
from allin1_infer.postprocessing import metrical
from allin1_infer.typings import AllInOneOutput


def _cfg():
  return Config(data=HarmonixConfig(), fps=100, best_threshold_downbeat=0.5)


def _logits():
  return AllInOneOutput(
    logits_beat=torch.tensor([[3.0, -1.0, 3.0, -1.0]]),
    logits_downbeat=torch.tensor([[3.0, -1.0, -1.0, 3.0]]),
    logits_section=torch.zeros(1, 4),
    logits_function=torch.zeros(1, 10, 4),
    embeddings=torch.zeros(1, 4, 2),
  )


def test_forwards_fast_viterbi_when_processor_signature_supports_it(monkeypatch):
  calls = []

  class SupportedProcessor:
    def __init__(self, beats_per_bar, threshold, fps, fast_viterbi=False):
      calls.append((beats_per_bar, threshold, fps, fast_viterbi))

    def __call__(self, activations):
      return np.array([[0.0, 1.0]])

  monkeypatch.setattr(metrical, 'DBNDownBeatTrackingProcessor', SupportedProcessor)

  result = metrical.postprocess_metrical_structure(_logits(), _cfg())

  assert calls == [([3, 4], 0.5, 100, True)]
  assert result == {'beats': [0.0], 'downbeats': [0.0], 'beat_positions': [1]}


def test_keeps_legacy_constructor_for_madmom_infer_020(monkeypatch):
  calls = []

  class LegacyProcessor:
    def __init__(self, beats_per_bar, threshold, fps):
      calls.append((beats_per_bar, threshold, fps))

    def __call__(self, activations):
      return np.array([[0.0, 1.0]])

  monkeypatch.setattr(metrical, 'DBNDownBeatTrackingProcessor', LegacyProcessor)

  result = metrical.postprocess_metrical_structure(_logits(), _cfg())

  assert calls == [([3, 4], 0.5, 100)]
  assert result == {'beats': [0.0], 'downbeats': [0.0], 'beat_positions': [1]}


def test_constructor_compatibility_does_not_change_decoder_output(monkeypatch):
  outputs = []

  class SupportedProcessor:
    def __init__(self, beats_per_bar, threshold, fps, fast_viterbi=False):
      pass

    def __call__(self, activations):
      outputs.append(activations.copy())
      return np.array([[0.0, 1.0], [0.5, 2.0]])

  monkeypatch.setattr(metrical, 'DBNDownBeatTrackingProcessor', SupportedProcessor)
  supported = metrical.postprocess_metrical_structure(_logits(), _cfg())

  class LegacyProcessor:
    def __init__(self, beats_per_bar, threshold, fps):
      pass

    def __call__(self, activations):
      outputs.append(activations.copy())
      return np.array([[0.0, 1.0], [0.5, 2.0]])

  monkeypatch.setattr(metrical, 'DBNDownBeatTrackingProcessor', LegacyProcessor)
  legacy = metrical.postprocess_metrical_structure(_logits(), _cfg())

  assert supported == legacy
  np.testing.assert_array_equal(outputs[0], outputs[1])
