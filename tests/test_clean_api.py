"""Fast contract tests for package-owned metadata and lifecycle semantics."""

import importlib

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from allin1_infer.checkpoints import checkpoint_metadata, load_checkpoints
from allin1_infer.session import AllInOneSession

analyze_module = importlib.import_module("allin1_infer.analyze")
stems_module = importlib.import_module("allin1_infer.stems")


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


def test_session_owns_both_components_and_loads_demucs_lazily(monkeypatch, tmp_path):
  class DummyModel:
    def parameters(self):
      return iter((torch.nn.Parameter(torch.zeros(1)),))
    def cpu(self):
      return self

  events = []

  class DummyDemucsProvider:
    def __init__(self, *, model_name, device, demucs_overlap, demucs_fp16):
      self.model_name = model_name
      self.device = device
      self.demucs_overlap = demucs_overlap
      self.demucs_fp16 = demucs_fp16
      self._loaded = False

    @property
    def model(self):
      self._loaded = True
      events.append(("demucs-load", self.model_name, self.device))
      return SimpleNamespace(sources=["bass", "drums", "other", "vocals"])

    @property
    def is_loaded(self):
      return self._loaded

    def release(self):
      events.append(("demucs-release", self.model_name))
      self._loaded = False

  monkeypatch.setattr("allin1_infer.session.load_pretrained_model", lambda **kwargs: DummyModel())
  monkeypatch.setattr("allin1_infer.session.DemucsProvider", DummyDemucsProvider)
  monkeypatch.setattr("allin1_infer.session.get_model_cache_dir", lambda: tmp_path)
  monkeypatch.setattr("allin1_infer.session.get_cache_size", lambda directory: 0.5)
  monkeypatch.setattr(
    "allin1_infer.session.list_cached_models",
    lambda directory: [{"name": "htdemucs.th", "path": str(tmp_path / "htdemucs.th")}],
  )
  monkeypatch.setattr(
    "allin1_infer.session.checkpoint_metadata",
    lambda model, path=None, overrides=None: {"model": model, "artifacts": []},
  )

  session = AllInOneSession(model="harmonix-fold0", device="cpu", demucs_model="htdemucs")
  session.load()
  assert session.status == "ready"
  assert session._model is not None
  info = session.cache_info()
  assert info["model"] == "harmonix-fold0"
  assert info["demucs_model"] == "htdemucs"
  assert info["model_loaded"] is True
  assert info["demucs_loaded"] is False
  assert info["components"] == [
    {"kind": "structure-model", "name": "harmonix-fold0", "loaded": True},
    {"kind": "separator", "name": "htdemucs", "loaded": False},
  ]
  assert not any(event[0] == "demucs-load" for event in events)

  session._demucs_provider.model
  assert session.cache_info()["demucs_loaded"] is True
  session.release()
  assert session.status == "released"
  assert ("demucs-load", "htdemucs", "cpu") in events
  assert ("demucs-release", "htdemucs") in events


def test_session_infer_reuses_owned_demucs_provider_in_both_mixed_input_paths(monkeypatch, tmp_path):
  provider = SimpleNamespace(
    device="cpu",
    demucs_overlap=0.25,
    demucs_fp16=False,
  )
  model = object()
  calls = []

  def fake_analyze(paths=None, **kwargs):
    calls.append((paths, kwargs))
    return SimpleNamespace(path=Path(paths))

  monkeypatch.setattr("allin1_infer.session.analyze", fake_analyze)

  session = AllInOneSession(device="cpu")
  session._model = model
  session._demucs_provider = provider
  session._status = "ready"

  session.infer(tmp_path / "track.wav")
  session.infer(tmp_path / "track.wav", keep_byproducts=True)

  assert len(calls) == 2
  for paths, kwargs in calls:
    assert paths == tmp_path / "track.wav"
    assert kwargs["_model_override"] is model
    assert kwargs["_default_stem_provider_override"] is provider
    assert kwargs["device"] == "cpu"
    assert kwargs["demucs_overlap"] == 0.25
    assert kwargs["demucs_fp16"] is False


def test_analyze_in_memory_and_cached_fallback_share_injected_demucs_provider(monkeypatch, tmp_path):
  track = tmp_path / "track.wav"
  track.write_bytes(b"fake")
  provider = object()
  route_calls = []

  monkeypatch.setattr(
    analyze_module,
    "separate_in_memory",
    lambda paths, demix_dir, device, demucs_overlap, demucs_fp16, provider=None: (
      route_calls.append(("separate_in_memory", provider)) or
      ([track], {track: demix_dir / "htdemucs" / track.stem}, {}, {})
    ),
  )
  monkeypatch.setattr(
    analyze_module,
    "get_stems",
    lambda paths, stems_dir, stem_provider=None, device="cpu", demucs_overlap=0.25, demucs_fp16=False: (
      route_calls.append(("get_stems", stem_provider)) or [stems_dir / "htdemucs" / track.stem]
    ),
  )
  monkeypatch.setattr(
    analyze_module,
    "extract_spectrograms",
    lambda demix_paths, spec_dir, multiprocess: [spec_dir / "track.npy"],
  )
  monkeypatch.setattr(
    analyze_module,
    "run_inference",
    lambda **kwargs: SimpleNamespace(path=kwargs["path"]),
  )

  result = analyze_module.analyze(
    track,
    device="cpu",
    multiprocess=False,
    _model_override=object(),
    _default_stem_provider_override=provider,
  )

  assert result.path == track
  assert route_calls == [
    ("separate_in_memory", provider),
    ("get_stems", provider),
  ]


def test_in_memory_separation_uses_injected_provider_configuration(monkeypatch, tmp_path):
  track = tmp_path / "track.wav"
  provider = SimpleNamespace(
    model=SimpleNamespace(sources=["bass", "drums", "other", "vocals"]),
    model_name="htdemucs",
    device="cuda:1",
    demucs_overlap=0.5,
    demucs_fp16=True,
  )
  calls = []

  monkeypatch.setattr(
    stems_module,
    "_run_demucs_separation",
    lambda model, path, device, overlap, fp16: (
      calls.append((model, path, device, overlap, fp16))
      or (torch.zeros(4, 2, 16), 44100)
    ),
  )
  monkeypatch.setattr(
    stems_module,
    "quantize_stem_to_madmom_mono_int16",
    lambda source: source.numpy(),
  )

  stems_module.separate_in_memory(
    [track],
    tmp_path / "demix",
    device="cpu",
    demucs_overlap=0.25,
    demucs_fp16=False,
    provider=provider,
  )

  assert calls == [(provider.model, track, "cuda:1", 0.5, True)]
