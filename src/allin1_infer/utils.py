"""Small path/JSON helpers shared across the package (path resolution,
result loading, compact JSON number arrays)."""

import re
import torch

from pathlib import Path
from .typings import PathLike, AnalysisResult


def resolve_device(device):
  """Resolve legacy automatic selection and validate explicit devices."""
  if device is None or device == 'auto':
    return 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cpu':
    return 'cpu'
  if not isinstance(device, str):
    raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
  if device == 'mps':
    mps = getattr(torch.backends, 'mps', None)
    if mps is None or not mps.is_available():
      raise RuntimeError("MPS requested but not available on this machine.")
    return 'mps'
  if device == 'cuda':
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA requested but not available on this machine.")
    return 'cuda'
  if not device.startswith('cuda:'):
    raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
  index_text = device[5:]
  if not index_text.isdigit():
    raise ValueError("CUDA device index must be a non-negative integer")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but not available on this machine.")
  if int(index_text) >= torch.cuda.device_count():
    raise RuntimeError(f"CUDA device index {index_text} is not available")
  return device


def compact_json_number_array(json_str: str):
  """Compact numbers (including floats) in JSON arrays to be on the same line."""
  return re.sub(
    r'(\[\n(?:\s*\d+(\.\d+)?,\n)+\s*\d+(\.\d+)?\n\s*\])',
    lambda m: m.group(1).replace('\n', '').replace(' ', ''),
    json_str
  )


def mkpath(path: PathLike):
  return Path(path).expanduser().resolve()


def load_result(
  path: PathLike,
  load_activations: bool = True,
  load_embeddings: bool = True,
) -> AnalysisResult:
  path = mkpath(path)
  result = AnalysisResult.from_json(
    path,
    load_activations=load_activations,
    load_embeddings=load_embeddings,
  )
  return result
