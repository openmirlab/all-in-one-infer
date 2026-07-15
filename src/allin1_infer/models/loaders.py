"""Downloads and builds pretrained AllInOne models (single fold or ensemble)
from the `taejunkim/allinone` Hugging Face Hub repo.

Each checkpoint embeds its own `Config` (restored via `OmegaConf.create`), so
the model architecture is reconstructed from the checkpoint itself rather
than from any config the caller passes in. For `harmonix-all`, the 8 fold
checkpoints are downloaded concurrently with a ThreadPoolExecutor -- but only
the download step, deliberately: `torch.load` + state_dict construction was
measured to regress under threading (GIL/CUDA-context contention outweighs
overlap), so model building stays sequential and `executor.map` preserves
fold order regardless of download completion order.

Reads: .allinone (AllInOne), .ensemble (Ensemble), ..typings (PathLike),
huggingface_hub (hf_hub_download), omegaconf
"""

import torch
import hashlib
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from .allinone import AllInOne
from .ensemble import Ensemble
from ..typings import PathLike
from ..checkpoints import checkpoint_metadata

NAME_TO_FILE = {
  'harmonix-fold0': 'harmonix-fold0-0vra4ys2.pth',
  'harmonix-fold1': 'harmonix-fold1-3ozjhtsj.pth',
  'harmonix-fold2': 'harmonix-fold2-gmgo0nsy.pth',
  'harmonix-fold3': 'harmonix-fold3-i92b7m8p.pth',
  'harmonix-fold4': 'harmonix-fold4-1bql5qo0.pth',
  'harmonix-fold5': 'harmonix-fold5-x4z5zeef.pth',
  'harmonix-fold6': 'harmonix-fold6-x7t226rq.pth',
  'harmonix-fold7': 'harmonix-fold7-qwwskhg6.pth',
}

ENSEMBLE_MODELS = {
  'harmonix-all': [
    'harmonix-fold0',
    'harmonix-fold1',
    'harmonix-fold2',
    'harmonix-fold3',
    'harmonix-fold4',
    'harmonix-fold5',
    'harmonix-fold6',
    'harmonix-fold7',
  ],
}


def _download_checkpoint(model_name: str, cache_dir: Optional[PathLike] = None,
                         *, checkpoint_url: Optional[str] = None,
                         checkpoint_sha256: Optional[str] = None,
                         checkpoint_config: Optional[PathLike] = None) -> str:
  filename = NAME_TO_FILE[model_name]
  metadata = checkpoint_metadata(model_name, path=checkpoint_config)
  artifact = next(a for a in metadata["artifacts"] if a.get("kind") == "checkpoint")
  url = checkpoint_url or artifact["url"]
  # The package-owned URL is the runtime source.  Keep the historical torch
  # hub/checkpoints default while allowing any host/path through overrides.
  import urllib.request
  if cache_dir is None:
    target_dir = Path(torch.hub.get_dir()) / "checkpoints"
  else:
    target_dir = Path(cache_dir)
  target_dir.mkdir(parents=True, exist_ok=True)
  path = target_dir / filename
  if not path.exists():
    urllib.request.urlretrieve(url, path)
  path = str(path)
  expected = (checkpoint_sha256 or artifact.get("sha256") or "").lower()
  if expected and expected != "0" * 64:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
      Path(path).unlink(missing_ok=True)
      raise ValueError(f"checkpoint SHA-256 mismatch for {model_name}")
  return str(path)


def _build_model_from_checkpoint(checkpoint_path: str, device) -> AllInOne:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  config = OmegaConf.create(checkpoint['config'])

  model = AllInOne(config).to(device)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  return model


def load_pretrained_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
  checkpoint_url: Optional[str] = None,
  checkpoint_sha256: Optional[str] = None,
  checkpoint_config: Optional[PathLike] = None,
):
  if model_name in ENSEMBLE_MODELS:
    return load_ensemble_model(model_name, cache_dir, device, checkpoint_config=checkpoint_config)

  model_name = model_name or list(NAME_TO_FILE.keys())[0]
  assert model_name in NAME_TO_FILE, f'Unknown model name: {model_name} (expected one of {list(NAME_TO_FILE.keys())})'

  if device is None:
    if torch.cuda.device_count():
      device = 'cuda'
    else:
      device = 'cpu'

  checkpoint_path = _download_checkpoint(model_name, cache_dir,
                                         checkpoint_url=checkpoint_url,
                                         checkpoint_sha256=checkpoint_sha256,
                                         checkpoint_config=checkpoint_config)
  return _build_model_from_checkpoint(checkpoint_path, device)


def load_ensemble_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
  *, checkpoint_config: Optional[PathLike] = None,
):
  fold_names = ENSEMBLE_MODELS[model_name]

  # Only hf_hub_download benefits from threading here: it's dominated by local
  # cache-resolution I/O (~0.2s/file even fully cached) that releases the GIL.
  # torch.load + state_dict construction do NOT benefit -- measured empirically,
  # threading those actually regresses wall-clock time (GIL/CUDA-context
  # contention across 8 threads outweighs any overlap). So only the download
  # step is parallelized; executor.map preserves fold order in its output
  # regardless of completion order, so ensemble fold order is unchanged.
  with ThreadPoolExecutor(max_workers=len(fold_names)) as executor:
    checkpoint_paths = list(executor.map(
      lambda name: _download_checkpoint(name, cache_dir, checkpoint_config=checkpoint_config),
      fold_names,
    ))

  if device is None:
    if torch.cuda.device_count():
      device = 'cuda'
    else:
      device = 'cpu'

  models = [_build_model_from_checkpoint(path, device) for path in checkpoint_paths]

  ensemble = Ensemble(models).to(device)
  ensemble.eval()

  return ensemble
