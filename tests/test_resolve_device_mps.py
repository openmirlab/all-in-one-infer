"""Regression tests: `mps` must be rejected outright by
`allin1_infer.utils.resolve_device` (org decision 2026-09-14, org canon
openmirlab-dev 5e588e6, art. 4b -- Apple MLX/MPS backends are permanently
out of scope for this org's projects).

Mirrors the sibling `mt3-infer` repo's `tests/test_framework.py` coverage
for its own `get_device()`: `mps` must raise `ValueError` unconditionally,
regardless of actual MPS availability, and `'auto'` must never resolve to
`mps` even when MPS is reported available.
"""

import pytest
import torch

from allin1_infer.utils import resolve_device


def test_resolve_device_rejects_mps_when_available(monkeypatch):
  mps = getattr(torch.backends, "mps", None)
  if mps is not None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
  with pytest.raises(ValueError, match="mps"):
    resolve_device("mps")


def test_resolve_device_rejects_mps_when_unavailable(monkeypatch):
  mps = getattr(torch.backends, "mps", None)
  if mps is not None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
  with pytest.raises(ValueError, match="mps"):
    resolve_device("mps")


def test_resolve_device_auto_never_resolves_to_mps_even_if_available(monkeypatch):
  """"auto" must never select mps, even when torch.backends.mps.is_available()
  would return True. The auto-detect path only ever considers CUDA-or-CPU."""
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  mps = getattr(torch.backends, "mps", None)
  if mps is not None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

  assert resolve_device("auto") == "cpu"
  assert resolve_device(None) == "cpu"
