# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Cache management additions)
# SPDX-License-Identifier: MIT

import numpy as np
import json
import torch

from dataclasses import asdict
from pathlib import Path
from glob import glob
from typing import List, Union
from .utils import mkpath, compact_json_number_array
from .typings import AllInOneOutput, AnalysisResult, PathLike
from .postprocessing import (
  postprocess_metrical_structure,
  postprocess_functional_structure,
  estimate_tempo_from_beats,
)


def run_inference(
  path: Path,
  spec_path: Path,
  model: torch.nn.Module,
  device: str,
  include_activations: bool,
  include_embeddings: bool,
) -> AnalysisResult:
  spec = np.load(spec_path)
  spec = torch.from_numpy(spec).unsqueeze(0).to(device)

  logits = model(spec)

  metrical_structure = postprocess_metrical_structure(logits, model.cfg)
  functional_structure = postprocess_functional_structure(logits, model.cfg)
  bpm = estimate_tempo_from_beats(metrical_structure['beats'])

  result = AnalysisResult(
    path=path,
    bpm=bpm,
    segments=functional_structure,
    **metrical_structure,
  )

  if include_activations:
    activations = compute_activations(logits)
    result.activations = activations

  if include_embeddings:
    result.embeddings = logits.embeddings[0].cpu().numpy()

  return result


def compute_activations(logits: AllInOneOutput):
  activations_beat = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
  activations_downbeat = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
  activations_segment = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
  activations_label = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()
  return {
    'beat': activations_beat,
    'downbeat': activations_downbeat,
    'segment': activations_segment,
    'label': activations_label,
  }


def expand_paths(paths: List[Path]):
  expanded_paths = set()
  for path in paths:
    if '*' in str(path) or '?' in str(path):
      matches = [Path(p) for p in glob(str(path))]
      if not matches:
        raise FileNotFoundError(f'Could not find any files matching {path}')
      expanded_paths.update(matches)
    else:
      expanded_paths.add(path)

  return sorted(expanded_paths)


def check_paths(paths: List[Path]):
  missing_files = []
  for path in paths:
    if not path.is_file():
      missing_files.append(str(path))
  if missing_files:
    raise FileNotFoundError(f'Could not find the following files: {missing_files}')


def rmdir_if_empty(path: Path):
  try:
    path.rmdir()
  except (FileNotFoundError, OSError):
    pass


def save_results(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = mkpath(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for result in results:
    out_path = out_dir / result.path.with_suffix('.json').name
    result = asdict(result)
    result['path'] = str(result['path'])

    activations = result.pop('activations')
    if activations is not None:
      np.savez(str(out_path.with_suffix('.activ.npz')), **activations)

    embeddings = result.pop('embeddings')
    if embeddings is not None:
      np.save(str(out_path.with_suffix('.embed.npy')), embeddings)

    json_str = json.dumps(result, indent=2)
    json_str = compact_json_number_array(json_str)
    out_path.with_suffix('.json').write_text(json_str)


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
