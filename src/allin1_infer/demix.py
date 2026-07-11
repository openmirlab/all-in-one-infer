# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Integration modifications)
# SPDX-License-Identifier: MIT

"""Legacy `demix()`/`demix_with_provider()` compatibility shims.

Both functions are thin pass-throughs to `stems.get_stems()`, kept only so
older call sites (and any external code written against the pre-stems.py
API) keep working. New code should call `stems.get_stems()` directly; nothing
here does real separation work anymore.

Reads: .stems (get_stems, DemucsProvider, StemProvider)
"""

import sys
import subprocess
import torch

from pathlib import Path
from typing import List, Union, Optional
from .stems import get_stems, DemucsProvider, StemProvider


def demix(
    paths: List[Path], demix_dir: Path, device: Union[str, torch.device],
    demucs_overlap: float = 0.25, demucs_fp16: bool = False,
):
  """
  Legacy demix function for backward compatibility.
  Now uses the new stems infrastructure.
  """
  return get_stems(paths, demix_dir, None, device, demucs_overlap, demucs_fp16)


def demix_with_provider(
    paths: List[Path], 
    demix_dir: Path, 
    stem_provider: Optional[StemProvider] = None,
    device: Union[str, torch.device] = 'cuda'
) -> List[Path]:
  """
  Demix audio files using specified stem provider.
  
  Parameters
  ----------
  paths : List[Path]
      List of audio file paths
  demix_dir : Path
      Directory to store stems
  stem_provider : Optional[StemProvider]
      Stem provider to use. If None, uses default DemucsProvider
  device : Union[str, torch.device]
      Device to use for separation
  
  Returns
  -------
  List[Path]
      List of paths to directories containing stems
  """
  return get_stems(paths, demix_dir, stem_provider, device)
