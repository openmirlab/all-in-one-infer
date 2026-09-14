"""Device-resolution regression tests for `allin1_infer.stems`.

Before this fix, `DemucsProvider.__init__`, `separate_in_memory()`, and the
module-level `get_stems()` all hardcoded a literal `device='cuda'` default
instead of routing through the package's single shared resolver
(`allin1_infer.utils.resolve_device`) the way `analyze()`/`session.py`/
`models/loaders.py` already do -- a direct call to any of these legacy entry
points without an explicit `device=` bypassed art. 4b's "auto"/explicit-
device contract entirely (no `'auto'` support, no validation, no raise on an
unavailable explicit device). `CustomSeparatorProvider.get_stems` separately
duplicated the resolver's cuda-if-available logic inline instead of reusing
it (a second, unreviewed device resolver).

These tests exercise `DemucsProvider` and `CustomSeparatorProvider` directly
(no real model/network work -- `DemucsProvider.model` is lazy, only touched
on first `.get_stems()`/`.model` access, never in `__init__`).
"""

from types import SimpleNamespace

import pytest
import torch

from allin1_infer.stems import DemucsProvider, CustomSeparatorProvider


def test_demucs_provider_default_is_auto_cuda_else_cpu(monkeypatch):
  for available in (True, False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda available=available: available)
    expected = "cuda" if available else "cpu"
    assert DemucsProvider().device == expected
    assert DemucsProvider(device="auto").device == expected


def test_demucs_provider_explicit_cpu_is_honored(monkeypatch):
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  assert DemucsProvider(device="cpu").device == "cpu"


def test_demucs_provider_invalid_device_string_raises():
  with pytest.raises(ValueError):
    DemucsProvider(device="metal")


def test_demucs_provider_unavailable_cuda_raises_rather_than_falling_back(monkeypatch):
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  with pytest.raises(RuntimeError, match="CUDA"):
    DemucsProvider(device="cuda")
  with pytest.raises(RuntimeError, match="CUDA"):
    DemucsProvider(device="cuda:0")


def test_custom_separator_provider_routes_through_shared_resolver(monkeypatch):
  """CustomSeparatorProvider.get_stems must reuse `utils.resolve_device`
  rather than a second, duplicated cuda-if-available check."""
  captured = {}

  class FakeSeparator:
    def separate(self, audio_path, output_dir, device):
      captured["device"] = device
      return output_dir

  for available in (True, False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda available=available: available)
    provider = CustomSeparatorProvider(FakeSeparator())
    provider.get_stems("track.wav", SimpleNamespace())
    assert captured["device"] == ("cuda" if available else "cpu")
