"""Ensemble fold registration and reproducible Demucs shift contracts.

`Ensemble` used to keep its folds in a plain list, so `.to()`, `.eval()` and
`.modules()` on the ensemble never reached them. Demucs' shift trick drew an
unseeded offset, so repeated runs of one file could differ (bpm 120 vs 122).
"""

import random

import torch
from omegaconf import OmegaConf
from torch import nn

from allin1_infer import stems
from allin1_infer.models.ensemble import Ensemble
from allin1_infer.typings import AllInOneOutput


class _Fold(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.cfg = OmegaConf.create({'best_threshold_beat': 0.2, 'best_threshold_downbeat': 0.4})
    self.scale = nn.Parameter(torch.tensor(float(scale)))
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    y = self.dropout(x) * self.scale
    return AllInOneOutput(logits_beat=y, logits_downbeat=y, logits_section=y,
                          logits_function=y, embeddings=y)


def test_ensemble_registers_folds_for_module_methods():
  folds = [_Fold(1.0), _Fold(3.0)]
  ensemble = Ensemble(folds)
  assert all(any(m is fold for m in ensemble.modules()) for fold in folds)
  ensemble.to(torch.float64)
  assert all(fold.scale.dtype == torch.float64 for fold in folds)
  ensemble.train()
  ensemble.eval()
  assert not any(fold.training for fold in folds)
  out = ensemble(torch.ones(3, dtype=torch.float64))
  assert torch.equal(out.logits_beat, torch.full((3,), 2.0, dtype=torch.float64))
  assert out.embeddings.shape == (3, 2)


def test_ensemble_fold_callables_replace_fold_calls_without_touching_modules():
  folds = [_Fold(1.0), _Fold(3.0)]
  ensemble = Ensemble(folds).eval()
  calls = []

  def wrap(fold):
    def call(x):
      calls.append(fold)
      return fold(x)
    return call

  ensemble.fold_callables = [wrap(fold) for fold in ensemble.models]
  ensemble(torch.ones(2))
  assert calls == folds
  assert list(ensemble.models) == folds


def test_demucs_separation_shift_is_seeded_and_restores_global_state(monkeypatch):
  def fake_apply_model(model, wav_batch, **kwargs):
    return torch.tensor([random.randint(0, 22050) for _ in range(4)])

  monkeypatch.setattr(stems, 'apply_model', fake_apply_model)
  random.seed(1234)
  expected_next = random.random()
  random.seed(1234)
  first = stems._apply_model_seeded(None, None, overlap=0.25)
  assert random.random() == expected_next
  second = stems._apply_model_seeded(None, None, overlap=0.25)
  assert torch.equal(first, second)
