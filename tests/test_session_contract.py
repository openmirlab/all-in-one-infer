"""Offline lifecycle and checkpoint-resolver contract tests."""

from types import SimpleNamespace

import pytest
import torch

from allin1_infer.session import AllInOneSession


class _Model:
  def __init__(self, device):
    self.device = device
    self.released = False

  def parameters(self):
    return iter((torch.nn.Parameter(torch.zeros(1)),))

  def cpu(self):
    self.released = True
    return self


class _Demucs:
  def __init__(self, *, model_name, device, demucs_overlap, demucs_fp16):
    self.device = device
    self.released = False
    self._loaded = False

  @property
  def is_loaded(self):
    return self._loaded

  def release(self):
    self.released = True


def test_session_load_reuses_then_rebuilds_and_close_is_terminal(monkeypatch, tmp_path):
  import allin1_infer.session as module

  built = []
  def load(**kwargs):
    model = _Model(kwargs["device"])
    built.append((kwargs, model))
    return model

  monkeypatch.setattr(module, "load_pretrained_model", load)
  monkeypatch.setattr(module, "DemucsProvider", _Demucs)
  session = AllInOneSession(model="harmonix-fold0", device="cuda:1")

  with pytest.raises(RuntimeError, match="ready"):
    session.infer(tmp_path / "audio.wav")
  session.load()
  session.load()
  assert len(built) == 1
  assert built[0][0]["device"] == "cuda:1"
  assert session._demucs_provider.device == "cuda:1"

  first_model, first_demucs = session._model, session._demucs_provider
  session.release()
  assert session.status == "released"
  assert first_model.released and first_demucs.released
  session.load()
  assert len(built) == 2

  session.close()
  session.close()
  assert session.status == "closed"
  with pytest.raises(RuntimeError, match="closed"):
    session.load()
  with pytest.raises(RuntimeError, match="ready"):
    session.infer(tmp_path / "audio.wav")


def test_failed_status_and_cache_info_share_custom_config_and_overrides(monkeypatch, tmp_path):
  import allin1_infer.session as module

  config = tmp_path / "checkpoints.toml"
  config.write_text(
    '[schema]\nversion = 1\n[models.harmonix-fold0]\n'
    '[[models.harmonix-fold0.artifacts]]\nkind = "checkpoint"\n'
    'url = "https://example.test/custom.pth"\nsha256 = "' + "a" * 64 + '"\n',
    encoding="utf-8",
  )
  seen = []
  monkeypatch.setattr(module, "load_pretrained_model", lambda **kwargs: seen.append(kwargs) or (_ for _ in ()).throw(ValueError("bad")))
  session = AllInOneSession(
    model="harmonix-fold0", cache_dir=tmp_path, checkpoint_config=config,
    checkpoint_overrides={"checkpoint_url": "https://example.test/override.pth"},
  )
  with pytest.raises(ValueError, match="bad"):
    session.load()
  assert session.status == "failed"
  assert seen[0]["checkpoint_config"] == config
  assert seen[0]["checkpoint_url"] == "https://example.test/override.pth"
  info = session.cache_info()
  assert info["checkpoint"]["checkpoint_url"] == "https://example.test/override.pth"
