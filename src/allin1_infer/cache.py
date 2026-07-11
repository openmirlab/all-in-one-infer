# Copyright (c) 2025 Bo-Yu Chen (Cache management additions)
# SPDX-License-Identifier: MIT

"""Demucs model-cache management for the CLI's `--cache-info`/`--clear-cache`
flags.

`get_model_cache_dir` points at torch hub's checkpoint directory (where
Demucs weights land); `get_cache_size`/`list_cached_models` inspect it,
`clear_model_cache` deletes `.pth`/`.th` files from it, and `print_cache_info`
is the human-readable report `cli.py --cache-info` prints. Unrelated to
per-track inference -- see `helpers.py` for that.

Reads: torch.hub
"""

from pathlib import Path
from typing import List


def get_model_cache_dir() -> Path:
  """
  Get the directory where separation models are cached.

  Returns
  -------
  Path
      Path to the model cache directory (torch hub checkpoints)
  """
  import torch
  torch_cache = Path(torch.hub.get_dir()) / 'checkpoints'
  return torch_cache


def get_cache_size(cache_dir: Path = None) -> float:
  """
  Get the total size of cached models in GB.

  Parameters
  ----------
  cache_dir : Path, optional
      Cache directory to check. If None, uses default model cache.

  Returns
  -------
  float
      Total cache size in GB
  """
  if cache_dir is None:
    cache_dir = get_model_cache_dir()

  if not cache_dir.exists():
    return 0.0

  total_size = 0
  for file in cache_dir.rglob('*'):
    if file.is_file():
      total_size += file.stat().st_size

  return total_size / (1024 ** 3)  # Convert to GB


def list_cached_models(cache_dir: Path = None) -> List[dict]:
  """
  List all cached separation models with their details.

  Parameters
  ----------
  cache_dir : Path, optional
      Cache directory to check. If None, uses default model cache.

  Returns
  -------
  List[dict]
      List of dicts with keys: 'name', 'size_mb', 'path', 'modified'
  """
  if cache_dir is None:
    cache_dir = get_model_cache_dir()

  if not cache_dir.exists():
    return []

  models = []
  # Check for both .pth and .th files (Demucs uses .th)
  for pattern in ['*.pth', '*.th']:
    for file in cache_dir.glob(pattern):
      size_mb = file.stat().st_size / (1024 ** 2)
      modified = file.stat().st_mtime

      models.append({
        'name': file.name,
        'size_mb': round(size_mb, 2),
        'path': str(file),
        'modified': modified
      })

  # Sort by modification time (newest first)
  models.sort(key=lambda x: x['modified'], reverse=True)

  return models


def clear_model_cache(cache_dir: Path = None, dry_run: bool = False) -> int:
  """
  Clear all cached separation models.

  Parameters
  ----------
  cache_dir : Path, optional
      Cache directory to clear. If None, uses default model cache.
  dry_run : bool, optional
      If True, only report what would be deleted without actually deleting

  Returns
  -------
  int
      Number of files removed (or would be removed if dry_run=True)
  """
  if cache_dir is None:
    cache_dir = get_model_cache_dir()

  if not cache_dir.exists():
    return 0

  count = 0
  # Only remove .pth and .th files (model checkpoints) to be safe
  for pattern in ['*.pth', '*.th']:
    for file in cache_dir.glob(pattern):
      if dry_run:
        print(f"Would remove: {file.name} ({file.stat().st_size / (1024**2):.2f} MB)")
      else:
        file.unlink()
        print(f"Removed: {file.name}")
      count += 1

  return count


def print_cache_info():
  """
  Print detailed information about the model cache.
  """
  cache_dir = get_model_cache_dir()
  total_size = get_cache_size(cache_dir)
  models = list_cached_models(cache_dir)

  print(f"\n{'='*60}")
  print(f"Model Cache Information")
  print(f"{'='*60}")
  print(f"Cache directory: {cache_dir}")
  print(f"Total size: {total_size:.2f} GB")
  print(f"Number of models: {len(models)}")
  print(f"\nCached models:")
  print(f"{'-'*60}")

  if not models:
    print("No cached models found")
  else:
    for model in models:
      from datetime import datetime
      modified_str = datetime.fromtimestamp(model['modified']).strftime('%Y-%m-%d %H:%M:%S')
      print(f"  {model['name']:<40} {model['size_mb']:>8.2f} MB  {modified_str}")

  print(f"{'='*60}\n")
