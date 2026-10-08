"""Ensemble of K AllInOne fold models: averages logits and thresholds.

`forward()` calls every fold model in a plain Python loop, then stacks and
means their logits/embeddings -- this is not fused or vectorized across
folds. That matters to callers doing torch.compile: compiling the Ensemble
wrapper itself just re-traces the Python loop, so analyze.py instead compiles
each fold individually and installs the wrapped callables in
`fold_callables` to guard against CUDA-graph buffer reuse across the loop
iterations (see analyze._wrap_compiled_fold). The folds live in an
`nn.ModuleList`, so `.to()`, `.eval()` and `.modules()` on the ensemble reach
every fold. The
per-fold best_threshold_beat/best_threshold_downbeat are averaged at
construction time into one shared `cfg` for the ensemble as a whole.

Reads: .allinone (AllInOne), ..typings (AllInOneOutput)
"""

import torch
import torch.nn as nn

from typing import Callable, List, Optional
from .allinone import AllInOne
from ..typings import AllInOneOutput


class Ensemble(nn.Module):
  def __init__(self, models: List[AllInOne]):
    super().__init__()

    cfg = models[0].cfg.copy()
    cfg.best_threshold_beat = sum([model.cfg.best_threshold_beat for model in models]) / len(models)
    cfg.best_threshold_downbeat = sum([model.cfg.best_threshold_downbeat for model in models]) / len(models)

    self.cfg = cfg
    self.models = nn.ModuleList(models)
    # Optional replacements for calling the folds (e.g. compiled wrappers);
    # None means call self.models directly.
    self.fold_callables: Optional[List[Callable]] = None

  def forward(self, x):
    folds = self.fold_callables if self.fold_callables is not None else self.models
    outputs: List[AllInOneOutput] = [fold(x) for fold in folds]
    avg = AllInOneOutput(
      logits_beat=torch.stack([output.logits_beat for output in outputs], dim=0).mean(dim=0),
      logits_downbeat=torch.stack([output.logits_downbeat for output in outputs], dim=0).mean(dim=0),
      logits_section=torch.stack([output.logits_section for output in outputs], dim=0).mean(dim=0),
      logits_function=torch.stack([output.logits_function for output in outputs], dim=0).mean(dim=0),
      embeddings=torch.stack([output.embeddings for output in outputs], dim=-1),
    )

    return avg
