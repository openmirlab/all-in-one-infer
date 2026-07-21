"""Explicit mixed-input lifecycle for All-In-One inference.

`AllInOneSession` owns both resident parts of the default pipeline: the
Harmonix structure model plus one reusable HTDemucs separator for mixed-input
calls. It keeps the strict load-before-infer contract while delegating actual
execution to the legacy `analyze()` pipeline, so the old one-shot API and the
direct-stems workflow remain backward compatible.

Reads: .analyze, .cache, .checkpoints, .models.loaders, .stems, torch
"""

from pathlib import Path
from typing import Any, Mapping, Optional

from .analyze import analyze
from .cache import get_model_cache_dir, get_cache_size, list_cached_models
from .checkpoints import checkpoint_metadata
from .models.loaders import load_pretrained_model
from .stems import DemucsProvider


class AllInOneSession:
  """Reusable, independently managed All-In-One pipeline session."""

  def __init__(self, *, model: str = "harmonix-all", device: Optional[str] = None,
               cache_dir: Optional[str | Path] = None,
               checkpoint_config: Optional[str | Path] = None,
               checkpoint_overrides: Optional[Mapping[str, Any]] = None,
               demucs_model: str = "htdemucs",
               demucs_overlap: float = 0.25,
               demucs_fp16: bool = False):
    self.model = model
    self.device = device
    self.cache_dir = Path(cache_dir) if cache_dir else None
    self.checkpoint_config = Path(checkpoint_config) if checkpoint_config else None
    self.checkpoint_overrides = dict(checkpoint_overrides or {})
    self.demucs_model = demucs_model
    self.demucs_overlap = demucs_overlap
    self.demucs_fp16 = demucs_fp16
    self._model = None
    self._demucs_provider: Optional[DemucsProvider] = None
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
      if self.device is None or self.device == "auto":
        self.device = str(next(self._model.parameters()).device)
      self._demucs_provider = DemucsProvider(
        model_name=self.demucs_model,
        device=self.device,
        demucs_overlap=self.demucs_overlap,
        demucs_fp16=self.demucs_fp16,
      )
      self._status = "ready"
      return self
    except Exception:
      self._model = None
      if self._demucs_provider is not None:
        self._demucs_provider.release()
      self._demucs_provider = None
      self._status = "failed"
      raise

  def infer(self, paths=None, **kwargs):
    if self._status != "ready" or self._model is None:
      raise RuntimeError("AllInOneSession must be ready; call load() before infer()")
    kwargs.setdefault("model", self.model)
    kwargs.setdefault("device", self.device)
    kwargs.setdefault("demucs_overlap", self.demucs_overlap)
    kwargs.setdefault("demucs_fp16", self.demucs_fp16)
    kwargs["_model_override"] = self._model
    kwargs["_default_stem_provider_override"] = self._demucs_provider
    return analyze(paths, **kwargs)

  def release(self) -> None:
    if self._demucs_provider is not None:
      self._demucs_provider.release()
      self._demucs_provider = None
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
    metadata = checkpoint_metadata(
      self.model, path=self.checkpoint_config, overrides=self.checkpoint_overrides,
    )
    return {
      "model": self.model,
      "demucs_model": self.demucs_model,
      "status": self._status,
      "model_loaded": self._model is not None,
      "demucs_loaded": self._demucs_provider is not None and self._demucs_provider.is_loaded,
      "cache_dir": str(directory),
      "cache_size_bytes": int(get_cache_size(directory) * (1024 ** 3)),
      "cached_models": list_cached_models(directory),
      "checkpoint": metadata,
      "components": [
        {
          "kind": "structure-model",
          "name": self.model,
          "loaded": self._model is not None,
        },
        {
          "kind": "separator",
          "name": self.demucs_model,
          "loaded": self._demucs_provider is not None and self._demucs_provider.is_loaded,
        },
      ],
    }

  def __enter__(self) -> "AllInOneSession":
    return self.load()

  def __exit__(self, exc_type, exc, tb) -> bool:
    self.close()
    return False


__all__ = ["AllInOneSession"]
