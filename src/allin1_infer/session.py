"""Explicit model lifecycle for All-In-One inference.

`AllInOneSession` owns checkpoint acquisition, model construction, device
placement, and release while delegating task execution to the legacy
`analyze()` pipeline.  The session is strict (load before infer); the legacy
one-shot function remains lazy and backward compatible.

Reads: .analyze, .cache, .checkpoints, .models.loaders, torch
"""

from pathlib import Path
from typing import Any, Mapping, Optional

from .analyze import analyze
from .cache import get_model_cache_dir, get_cache_size, list_cached_models
from .checkpoints import checkpoint_metadata
from .models.loaders import load_pretrained_model


class AllInOneSession:
  """Reusable, independently managed All-In-One model session."""

  def __init__(self, *, model: str = "harmonix-all", device: Optional[str] = None,
               cache_dir: Optional[str | Path] = None,
               checkpoint_config: Optional[str | Path] = None,
               checkpoint_overrides: Optional[Mapping[str, Any]] = None):
    self.model = model
    self.device = device
    self.cache_dir = Path(cache_dir) if cache_dir else None
    self.checkpoint_config = Path(checkpoint_config) if checkpoint_config else None
    self.checkpoint_overrides = dict(checkpoint_overrides or {})
    self._model = None
    self._status = "new"

  @property
  def status(self) -> str:
    return self._status

  def load(self) -> "AllInOneSession":
    if self._status == "closed":
      raise RuntimeError("cannot load a closed AllInOneSession")
    if self._status == "ready":
      return self
    self._status = "loading"
    try:
      kwargs = dict(self.checkpoint_overrides)
      if self.checkpoint_config is not None:
        kwargs["checkpoint_config"] = self.checkpoint_config
      self._model = load_pretrained_model(
        model_name=self.model,
        cache_dir=self.cache_dir,
        device=self.device,
        **kwargs,
      )
      if self.device is None:
        self.device = str(next(self._model.parameters()).device)
      self._status = "ready"
      return self
    except Exception:
      self._model = None
      self._status = "failed"
      raise

  def infer(self, paths=None, **kwargs):
    if self._status != "ready" or self._model is None:
      raise RuntimeError("AllInOneSession must be ready; call load() before infer()")
    kwargs.setdefault("model", self.model)
    kwargs.setdefault("device", self.device)
    kwargs["_model_override"] = self._model
    return analyze(paths, **kwargs)

  def release(self) -> None:
    if self._model is not None:
      if hasattr(self._model, "cpu"):
        self._model.cpu()
      self._model = None
    try:
      import torch
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except ImportError:
      pass
    if self._status != "closed":
      self._status = "released"

  def close(self) -> None:
    self.release()
    self._status = "closed"

  def cache_info(self) -> dict[str, Any]:
    directory = self.cache_dir or get_model_cache_dir()
    metadata = checkpoint_metadata(self.model, path=self.checkpoint_config)
    return {
      "model": self.model,
      "status": self._status,
      "model_loaded": self._model is not None,
      "cache_dir": str(directory),
      "cache_size_bytes": int(get_cache_size(directory) * (1024 ** 3)),
      "cached_models": list_cached_models(directory),
      "checkpoint": metadata,
    }

  def __enter__(self) -> "AllInOneSession":
    return self.load()

  def __exit__(self, exc_type, exc, tb) -> bool:
    self.close()
    return False


__all__ = ["AllInOneSession"]
