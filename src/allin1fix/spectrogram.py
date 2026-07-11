import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

STEM_NAMES = ['bass', 'drums', 'other', 'vocals']


def build_spec_processor() -> SequentialProcessor:
  """Pre-processing chain, copied from madmom. Shared by the disk-based and
  in-memory extraction paths so both stay numerically identical."""
  frames = FramedSignalProcessor(
    frame_size=2048,
    fps=int(44100 / 441)
  )
  stft = ShortTimeFourierTransformProcessor()  # caching FFT window
  filt = FilteredSpectrogramProcessor(
    num_bands=12,
    fmin=30,
    fmax=17000,
    norm_filters=True
  )
  spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
  return SequentialProcessor([frames, stft, filt, spec])


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path, multiprocess: bool = True):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    processor = build_spec_processor()

    # Process all tracks using multiprocessing.
    if multiprocess:
      pool = Pool()
      map_fn = pool.imap
    else:
      pool = None
      map_fn = map

    iterator = map_fn(_extract_spectrogram, [
      (src, dst, processor)
      for src, dst in todos
    ])
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass

    if pool:
      pool.close()
      pool.join()

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor]):
  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)


def compute_spectrogram_from_stem_arrays(
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    processor: SequentialProcessor = None,
) -> np.ndarray:
  """In-memory counterpart to _extract_spectrogram's per-file Signal() reads.

  `stems` must hold the same int16 mono arrays that madmom's
  Signal(wav_path, num_channels=1) would read back from on-disk stem wavs
  (see stems.quantize_stem_to_madmom_mono_int16) -- given those, this is
  bit-for-bit identical to the disk-based path.
  """
  processor = processor or build_spec_processor()
  specs = [processor(Signal(stems[name], sample_rate=sample_rate)) for name in STEM_NAMES]
  return np.stack(specs)  # instruments, frames, bins


def extract_spectrograms_from_arrays(
    arrays_by_path: Dict[Path, Dict[str, np.ndarray]],
    spec_dir: Path,
    sample_rate: int,
) -> Dict[Path, Path]:
  """In-memory counterpart to extract_spectrograms(): skips reading stem wavs
  from disk. Always sequential -- routing these already-in-memory arrays
  through a multiprocessing.Pool would reintroduce most of the pickling/IPC
  cost that this path exists to avoid.
  """
  processor = build_spec_processor()
  spec_paths = {}
  for path, stems in tqdm(arrays_by_path.items(), desc='Extracting spectrograms (in-memory)'):
    dst = spec_dir / f'{path.stem}.npy'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file():
      spec = compute_spectrogram_from_stem_arrays(stems, sample_rate, processor)
      np.save(str(dst), spec)
    spec_paths[path] = dst
  return spec_paths
