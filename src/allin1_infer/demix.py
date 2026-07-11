# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Integration modifications)
# SPDX-License-Identifier: MIT

"""`demix()`: analyze()'s entry point into the default (disk-based, no custom
provider) HTDemucs separation path.

A thin pass-through to `stems.get_stems()` with `stem_provider=None`, kept as
its own function/module because analyze.py calls it specifically for the
keep_byproducts branch (see analyze._select_stems).

Reads: .stems (get_stems)
"""

import torch

from pathlib import Path
from typing import List, Union
from .stems import get_stems


def demix(
    paths: List[Path], demix_dir: Path, device: Union[str, torch.device],
    demucs_overlap: float = 0.25, demucs_fp16: bool = False,
):
  """
  Demix audio files using the default DemucsProvider.
  """
  return get_stems(paths, demix_dir, None, device, demucs_overlap, demucs_fp16)
