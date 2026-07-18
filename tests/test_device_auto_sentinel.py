"""Regression tests for the explicit `"auto"` device-string sentinel.

`device="auto"` must resolve identically to leaving `device` unset (None, or
an already-baked-in `cuda`-if-available default) at every device-resolution
site, while explicit literal devices ("cpu", "cuda:0", ...) must keep passing
through unchanged. See `allin1_infer.utils.resolve_device`, the single shared
helper all these sites now call through.
"""

import importlib
from types import SimpleNamespace

import torch

from allin1_infer.utils import resolve_device
from allin1_infer.session import AllInOneSession

loaders_module = importlib.import_module("allin1_infer.models.loaders")
analyze_module = importlib.import_module("allin1_infer.analyze")


# ---------------------------------------------------------------------------
# resolve_device itself
# ---------------------------------------------------------------------------

def test_resolve_device_none_and_auto_agree(monkeypatch):
  for available in (True, False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda available=available: available)
    expected = "cuda" if available else "cpu"
    assert resolve_device(None) == expected
    assert resolve_device("auto") == expected


def test_resolve_device_passes_through_explicit_devices(monkeypatch):
  # Explicit devices must be untouched regardless of what CUDA reports.
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  assert resolve_device("cpu") == "cpu"
  assert resolve_device("cuda") == "cuda"
  assert resolve_device("cuda:0") == "cuda:0"


# ---------------------------------------------------------------------------
# loaders.py: load_pretrained_model / load_ensemble_model
# ---------------------------------------------------------------------------

def test_load_pretrained_model_auto_matches_unset(monkeypatch):
  captured = []
  monkeypatch.setattr(loaders_module, "_download_checkpoint", lambda *a, **k: "checkpoint.pth")
  monkeypatch.setattr(
    loaders_module,
    "_build_model_from_checkpoint",
    lambda checkpoint_path, device: captured.append(device) or SimpleNamespace(device=device),
  )
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  loaders_module.load_pretrained_model(model_name="harmonix-fold0", device=None)
  loaders_module.load_pretrained_model(model_name="harmonix-fold0", device="auto")

  assert captured == ["cpu", "cpu"]


def test_load_pretrained_model_explicit_device_passes_through(monkeypatch):
  monkeypatch.setattr(loaders_module, "_download_checkpoint", lambda *a, **k: "checkpoint.pth")
  captured = []
  monkeypatch.setattr(
    loaders_module,
    "_build_model_from_checkpoint",
    lambda checkpoint_path, device: captured.append(device) or SimpleNamespace(device=device),
  )
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

  loaders_module.load_pretrained_model(model_name="harmonix-fold0", device="cpu")

  assert captured == ["cpu"]


def test_load_ensemble_model_auto_matches_unset(monkeypatch):
  monkeypatch.setattr(loaders_module, "_download_checkpoint", lambda *a, **k: "checkpoint.pth")
  captured = []
  monkeypatch.setattr(
    loaders_module,
    "_build_model_from_checkpoint",
    lambda checkpoint_path, device: captured.append(device) or SimpleNamespace(device=device),
  )
  monkeypatch.setattr(loaders_module, "Ensemble", lambda models: SimpleNamespace(
    models=models, to=lambda device: SimpleNamespace(eval=lambda: None),
  ))
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  loaders_module.load_ensemble_model(model_name="harmonix-all", device=None)
  loaders_module.load_ensemble_model(model_name="harmonix-all", device="auto")

  # 8 folds x 2 calls, all resolved to 'cpu'.
  assert captured == ["cpu"] * 16


# ---------------------------------------------------------------------------
# analyze.py: device="auto" resolved at entry, before anything inspects it
# ---------------------------------------------------------------------------

def _patch_minimal_analyze_pipeline(monkeypatch, tmp_path, track):
  """Stubs everything analyze() touches downstream of device-resolution so a
  single-track run completes without real audio/model work, mirroring the
  pattern in test_clean_api.py's analyze-level tests."""
  monkeypatch.setattr(
    analyze_module,
    "_select_stems",
    lambda todo_paths, demix_dir, spec_dir, device, *a, **k: ([demix_dir / "htdemucs" / track.stem], []),
  )
  monkeypatch.setattr(
    analyze_module,
    "extract_spectrograms",
    lambda demix_paths, spec_dir, multiprocess: [spec_dir / "track.npy"],
  )
  captured_devices = []

  def fake_run_inference(**kwargs):
    captured_devices.append(kwargs["device"])
    return SimpleNamespace(path=kwargs["path"])

  monkeypatch.setattr(analyze_module, "run_inference", fake_run_inference)
  return captured_devices


def test_analyze_auto_device_matches_unset(monkeypatch, tmp_path):
  track = tmp_path / "track.wav"
  track.write_bytes(b"fake")
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  captured = _patch_minimal_analyze_pipeline(monkeypatch, tmp_path, track)

  # "unset" is simulated the same way analyze()'s own signature default is
  # baked in: passing the already-resolved concrete value.
  unset_equivalent = resolve_device(None)
  analyze_module.analyze(
    track,
    device=unset_equivalent,
    multiprocess=False,
    _model_override=object(),
  )
  analyze_module.analyze(
    track,
    device="auto",
    multiprocess=False,
    _model_override=object(),
  )

  assert captured == [unset_equivalent, unset_equivalent]


def test_analyze_explicit_device_passes_through(monkeypatch, tmp_path):
  track = tmp_path / "track.wav"
  track.write_bytes(b"fake")

  captured = _patch_minimal_analyze_pipeline(monkeypatch, tmp_path, track)

  analyze_module.analyze(
    track,
    device="cpu",
    multiprocess=False,
    _model_override=object(),
  )

  assert captured == ["cpu"]


# ---------------------------------------------------------------------------
# session.py: AllInOneSession(device="auto") must back-propagate a concrete
# resolved device the same way the unset (None) case does (line ~64), so
# later DemucsProvider/infer() calls never see the literal "auto" string.
# ---------------------------------------------------------------------------

def _install_session_doubles(monkeypatch, tmp_path, resolved_param_device):
  class DummyModel:
    def parameters(self):
      return iter((torch.nn.Parameter(torch.zeros(1), requires_grad=False).to(resolved_param_device),))

    def cpu(self):
      return self

  demucs_events = []

  class DummyDemucsProvider:
    def __init__(self, *, model_name, device, demucs_overlap, demucs_fp16):
      self.model_name = model_name
      self.device = device
      demucs_events.append(("demucs-init", device))

    @property
    def is_loaded(self):
      return False

    def release(self):
      pass

  monkeypatch.setattr("allin1_infer.session.load_pretrained_model", lambda **kwargs: DummyModel())
  monkeypatch.setattr("allin1_infer.session.DemucsProvider", DummyDemucsProvider)
  monkeypatch.setattr("allin1_infer.session.get_model_cache_dir", lambda: tmp_path)
  monkeypatch.setattr("allin1_infer.session.get_cache_size", lambda directory: 0.5)
  monkeypatch.setattr("allin1_infer.session.list_cached_models", lambda directory: [])
  monkeypatch.setattr(
    "allin1_infer.session.checkpoint_metadata",
    lambda model, path=None: {"model": model, "artifacts": []},
  )
  return demucs_events


def test_session_auto_device_back_propagates_like_unset(monkeypatch, tmp_path):
  # A model that ended up placed on "cpu" (as if load_pretrained_model had
  # already resolved "auto"/None internally).
  demucs_events = _install_session_doubles(monkeypatch, tmp_path, "cpu")

  session = AllInOneSession(model="harmonix-fold0", device="auto", demucs_model="htdemucs")
  session.load()

  # The literal "auto" string must NOT survive into self.device or get
  # forwarded to DemucsProvider -- it must resolve to the concrete device
  # the model actually landed on, exactly like the device=None case does.
  assert session.device == "cpu"
  assert demucs_events == [("demucs-init", "cpu")]


def test_session_unset_device_back_propagates(monkeypatch, tmp_path):
  demucs_events = _install_session_doubles(monkeypatch, tmp_path, "cpu")

  session = AllInOneSession(model="harmonix-fold0", device=None, demucs_model="htdemucs")
  session.load()

  assert session.device == "cpu"
  assert demucs_events == [("demucs-init", "cpu")]


def test_session_explicit_device_is_never_reresolved(monkeypatch, tmp_path):
  # Deliberately place the dummy model's parameters on a DIFFERENT device
  # than the explicit input, so that if the `self.device is None or
  # self.device == "auto"` guard were ever accidentally dropped (falling
  # back to always re-resolving from `model.parameters()`), this test would
  # catch it -- an explicit "cpu" input must never be overwritten with
  # whatever device the (stubbed) model reports.
  demucs_events = _install_session_doubles(monkeypatch, tmp_path, "cuda:0")

  session = AllInOneSession(model="harmonix-fold0", device="cpu", demucs_model="htdemucs")
  session.load()

  assert session.device == "cpu"
  assert demucs_events == [("demucs-init", "cpu")]
