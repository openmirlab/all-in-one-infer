"""Package-owned checkpoint metadata and validation.

The runtime reads this release-pinned TOML file instead of scattering model
URLs through loaders.  Callers may supply a custom config path or metadata
mapping without naming a particular hosting service.

Reads: pathlib, tomllib, package-local config/checkpoints.toml
"""

from pathlib import Path
from typing import Any, Mapping
try:
  import tomllib
except ModuleNotFoundError:  # Python 3.10
  import tomli as tomllib


def checkpoint_config_path() -> Path:
  return Path(__file__).with_name("config") / "checkpoints.toml"


def load_checkpoints(path: Path | str | None = None) -> dict[str, Any]:
  source = Path(path) if path is not None else checkpoint_config_path()
  try:
    with source.open("rb") as handle:
      data = tomllib.load(handle)
  except (OSError, tomllib.TOMLDecodeError) as exc:
    raise ValueError(f"invalid checkpoint config: {source}") from exc
  if data.get("schema", {}).get("version") != 1 or not isinstance(data.get("models"), dict):
    raise ValueError("checkpoint config must define schema.version=1 and models")
  for key, model in data["models"].items():
    artifacts = model.get("artifacts") if isinstance(model, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
      raise ValueError(f"invalid checkpoint metadata for {key}")
    for artifact in artifacts:
      digest = artifact.get("sha256") if isinstance(artifact, dict) else None
      if not isinstance(artifact, dict) or not str(artifact.get("url", "")).startswith("https://"):
        raise ValueError(f"invalid artifact URL for {key}")
      if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError(f"invalid artifact SHA-256 for {key}")
  return data


def checkpoint_metadata(model: str, *, path: Path | str | None = None,
                        overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
  try:
    value = dict(load_checkpoints(path)["models"][model])
  except KeyError as exc:
    raise KeyError(f"unknown checkpoint model: {model}") from exc
  if overrides:
    value.update(overrides)
  return value


__all__ = ["checkpoint_config_path", "load_checkpoints", "checkpoint_metadata"]
