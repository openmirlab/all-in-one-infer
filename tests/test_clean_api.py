"""Fast contract tests for package-owned metadata and lifecycle semantics."""

import pytest

from allin1_infer.checkpoints import checkpoint_metadata, load_checkpoints
from allin1_infer.session import AllInOneSession


def test_checkpoint_config_is_package_owned_and_complete():
  data = load_checkpoints()
  assert set(data["models"]) >= {"harmonix-all", "harmonix-fold0", "harmonix-fold7"}
  artifact = checkpoint_metadata("harmonix-fold0")["artifacts"][0]
  assert artifact["url"].startswith("https://")
  assert len(artifact["sha256"]) == 64


def test_session_is_strict_and_close_idempotent():
  session = AllInOneSession()
  assert session.status == "new"
  with pytest.raises(RuntimeError):
    session.infer("track.wav")
  session.close()
  session.close()
  assert session.status == "closed"
  with pytest.raises(RuntimeError):
    session.load()


def test_session_loads_injected_model(monkeypatch):
  class DummyModel:
    def parameters(self):
      return iter(())
    def cpu(self):
      return self

  monkeypatch.setattr("allin1_infer.session.load_pretrained_model", lambda **kwargs: DummyModel())
  session = AllInOneSession(model="harmonix-fold0", device="cpu")
  session.load()
  assert session.status == "ready"
  assert session._model is not None
  session.release()
  assert session.status == "released"
