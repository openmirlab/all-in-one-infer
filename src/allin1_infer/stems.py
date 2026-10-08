# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Integration modifications)
# SPDX-License-Identifier: MIT

"""Stem input/output handling for flexible source separation.

This module provides interfaces for working with pre-separated stems,
allowing developers to use custom source separation models or skip source
separation entirely if stems are already available. It also holds the
in-memory vs. disk-based separation split: `DemucsProvider.get_stems` writes
stem wavs to disk and is used whenever byproducts must be kept or stems
already exist on disk, while `separate_in_memory` skips that write/read round
trip (~0.83s/track) by handing stem tensors straight to spectrogram
extraction. The two paths must stay numerically identical, which is what
`quantize_stem_to_madmom_mono_int16` guarantees: it reproduces, bit-for-bit,
the int16 mono array madmom's `Signal(path, num_channels=1)` would read back
from a wav written by `demucs_infer.audio.save_audio` -- if demucs-infer ever
changes `save_audio`'s defaults (clip mode, bit depth), this needs
re-verification against the real on-disk round trip.

`_run_demucs_separation` loads the user's ORIGINAL input file (whatever
format they passed in -- wav/flac/mp3/...), not demucs's own output; see
`_load_input_audio`'s docstring for the wav/flac-via-soundfile,
lossy-via-ffmpeg split. This package never imports torchaudio (issue #7):
demucs-infer dropped it at ffe0080, so it is no longer installed
transitively.

Reads: demucs_infer (pretrained.get_model, apply.apply_model,
audio.AudioFile, audio.save_audio)
"""

import random
import subprocess

import torch
import soundfile as sf
from pathlib import Path
from typing import List, Union, Optional, Dict, Callable, Protocol, Tuple
from abc import ABC, abstractmethod

# Import demucs-infer for source separation
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import AudioFile, save_audio, prevent_clip

import numpy as np

from .spectrogram import STEM_NAMES
from .utils import resolve_device


class StemSeparator(Protocol):
    """Protocol for custom source separation implementations."""
    
    def separate(self, audio_path: Path, output_dir: Path, device: Union[str, torch.device]) -> Path:
        """
        Separate audio into stems.
        
        Parameters
        ----------
        audio_path : Path
            Path to the input audio file
        output_dir : Path
            Directory to save separated stems
        device : Union[str, torch.device]
            Device to use for separation
            
        Returns
        -------
        Path
            Path to directory containing separated stems (bass.wav, drums.wav, other.wav, vocals.wav)
        """
        ...


class StemProvider(ABC):
    """Abstract base class for stem providers."""
    
    @abstractmethod
    def get_stems(self, identifier: Union[Path, str], output_dir: Path) -> Path:
        """
        Get stems for a given audio identifier.
        
        Parameters
        ----------
        identifier : Union[Path, str]
            Audio file path or identifier
        output_dir : Path
            Directory to save/copy stems
            
        Returns
        -------
        Path
            Path to directory containing stems (bass.wav, drums.wav, other.wav, vocals.wav)
        """
        pass


# Extensions decoded via soundfile. Verified bit-identical to the previous
# torchaudio==2.7.1 decode (soundfile==0.13.1 -- PCM16/24/32 wav + FLAC,
# mono/stereo synthetic fixtures, plus the two real multi-minute stereo wav
# assets under assets/ -- np.array_equal exact on every file; see
# CHANGELOG.md's 3.1.0 entry). Everything else (mp3 in particular) goes
# through ffmpeg instead: soundfile's mp3 decoder (libmpg123) measured up to
# ~2.4e-6 per sample away from the ffmpeg-backed decode, and the README's
# "Concerning MP3 Files" section documents decoder-dependent offsets, so
# lossy formats never silently switch to soundfile. (demucs-infer's own
# Separator does fall back to soundfile for mp3; this package deliberately
# does not.)
_LOSSLESS_SOUNDFILE_EXTS = {'.wav', '.flac'}


def _load_input_audio(audio_path: Union[Path, str]) -> Tuple[torch.Tensor, int]:
    """Load the user's ORIGINAL input audio file (arbitrary format --
    wav/flac/mp3/whatever ffmpeg can decode -- NOT demucs's own stem output)
    into a [channels, samples] float32 tensor at the file's native sample
    rate and channel count.

    wav/flac go through soundfile (see `_LOSSLESS_SOUNDFILE_EXTS`), so they
    need no external executable. Every other format goes through
    demucs-infer's `AudioFile` (the ffprobe/ffmpeg executables). That path
    was measured bit-exact against the torchaudio==2.7.1 ffmpeg-backend
    decode it replaces (ffmpeg 6.1.1; mp3/ogg/m4a, mono and stereo,
    22.05/44.1/48 kHz), so existing mp3 results don't change. If ffmpeg is
    missing or can't decode the file, a clear actionable error is raised
    instead of falling back to another decoder.
    """
    path = Path(audio_path)
    if path.suffix.lower() in _LOSSLESS_SOUNDFILE_EXTS:
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr

    try:
        audio_file = AudioFile(path)
        wav = audio_file.read(streams=0)
        return wav, audio_file.samplerate()
    except FileNotFoundError as err:
        reason = (
            f"the ffmpeg/ffprobe executables were not found ({err}). "
            "Install FFmpeg so both are on PATH"
        )
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as err:
        reason = f"ffmpeg could not decode it ({type(err).__name__}: {err})"
    raise RuntimeError(
        f"Failed to load '{path}': {reason}. Formats other than wav/flac "
        "(mp3 included) are decoded only via ffmpeg -- never silently via "
        "soundfile, whose decode differs -- so either make ffmpeg available "
        "or convert the file to wav/flac, which this package decodes via "
        "soundfile without ffmpeg."
    )


# demucs_infer.apply.apply_model's "shift trick" (shifts=1 by default) offsets
# the mix by up to 0.5 s chosen with the stdlib `random` module, so unseeded
# runs of the same file differ slightly (observed: the same track's bpm
# alternating between 120 and 122). Seeding it for each separation makes
# results reproducible; each run is still one ordinary draw of the shift,
# as upstream intends. The caller's global random state is restored after.
_DEMUCS_SHIFT_SEED = 0


def _apply_model_seeded(model, wav_batch, **kwargs) -> torch.Tensor:
    state = random.getstate()
    random.seed(_DEMUCS_SHIFT_SEED)
    try:
        return apply_model(model, wav_batch, **kwargs)
    finally:
        random.setstate(state)


def _run_demucs_separation(
    model,
    audio_path: Union[Path, str],
    device: Union[str, torch.device],
    overlap: float,
    fp16: bool,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[torch.Tensor, int]:
    """Shared load -> force-stereo -> apply_model -> to-CPU sequence used by
    both DemucsProvider.get_stems (writes stems to disk) and
    separate_in_memory (keeps stems as in-memory arrays), so the two paths
    can't numerically drift apart. Returns (sources, sr) with the batch
    dimension already squeezed out and sources moved to CPU.
    """
    if progress_callback:
        progress_callback("Loading audio file", 0.2)

    wav, sr = _load_input_audio(audio_path)

    # Ensure stereo (demucs requires 2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Add batch dimension and move to device
    wav_batch = wav.unsqueeze(0).to(device)

    if progress_callback:
        progress_callback("Separating audio sources", 0.3)

    with torch.no_grad():
        if fp16 and 'cuda' in str(device):
            with torch.autocast('cuda', dtype=torch.float16):
                sources = _apply_model_seeded(
                    model, wav_batch, device=device,
                    progress=bool(progress_callback), overlap=overlap,
                )
        else:
            sources = _apply_model_seeded(
                model, wav_batch, device=device,
                progress=bool(progress_callback), overlap=overlap,
            )

    # Move to CPU immediately to free GPU memory, remove batch dimension
    sources = sources.cpu().squeeze(0)

    del wav_batch
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sources, sr


class DemucsProvider(StemProvider):
    """Default stem provider using integrated separation module with model caching."""

    def __init__(
        self,
        model_name: str = 'htdemucs',
        device: Union[str, torch.device] = 'auto',
        demucs_overlap: float = 0.25,
        demucs_fp16: bool = False,
    ):
        self.model_name = model_name
        # Route through the package's single device resolver (see
        # `allin1_infer.utils.resolve_device`) instead of a hardcoded literal
        # default -- 'auto' resolves to cuda-if-available/else-cpu exactly
        # like every other entry point, an explicit unavailable/invalid
        # device raises rather than silently landing wherever `.to()` would
        # put it, and a caller who already resolved a concrete device
        # upstream (e.g. `analyze()`) re-resolves to the same value.
        self.device = resolve_device(device)
        # EXPERIMENTAL, accuracy-affecting knobs -- defaults reproduce prior
        # behavior exactly (0.25 is demucs' own apply_model default, fp16=False
        # keeps separation in fp32). See profiling notes: non-default overlap
        # and fp16 autocast can shift segment boundaries slightly.
        self.demucs_overlap = demucs_overlap
        self.demucs_fp16 = demucs_fp16
        self._model = None  # Cache for loaded model

    @property
    def model(self):
        """Lazy-load and cache the separation model."""
        if self._model is None:
            self._model = get_model(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()  # Freeze batch norm, dropout for inference
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def clear_model_cache(self):
        """Clear cached model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self):
        """Release the resident separation model without touching checkpoint files."""
        self.clear_model_cache()

    def get_stems(
        self,
        identifier: Union[Path, str],
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Path:
        """
        Use integrated separation module to separate audio into stems.

        Parameters
        ----------
        identifier : Union[Path, str]
            Path to audio file to separate
        output_dir : Path
            Directory to save separated stems
        progress_callback : Optional[Callable[[str, float], None]]
            Optional callback function(message: str, progress: float [0-1])

        Returns
        -------
        Path
            Directory containing separated stems
        """
        audio_path = Path(identifier)
        stems_dir = output_dir / self.model_name / audio_path.stem

        # Check if stems already exist
        required_stems = [f'{name}.wav' for name in STEM_NAMES]
        if all((stems_dir / stem).exists() for stem in required_stems):
            if progress_callback:
                progress_callback("Stems already exist, skipping separation", 1.0)
            return stems_dir

        # Create output directory
        stems_dir.mkdir(parents=True, exist_ok=True)

        # Load model (cached after first call)
        if progress_callback:
            progress_callback("Loading separation model", 0.1)

        model = self.model  # Uses cached model if available

        sources, sr = _run_demucs_separation(
            model, audio_path, self.device, self.demucs_overlap, self.demucs_fp16,
            progress_callback=progress_callback,
        )

        # Save stems
        if progress_callback:
            progress_callback("Saving separated stems", 0.8)

        source_names = model.sources
        for i, source_name in enumerate(source_names):
            stem_path = stems_dir / f'{source_name}.wav'
            save_audio(sources[i], str(stem_path), sr)

        # Clean up CPU memory
        del sources

        if progress_callback:
            progress_callback("Separation complete", 1.0)

        return stems_dir


def quantize_stem_to_madmom_mono_int16(wav: torch.Tensor) -> np.ndarray:
    """Reproduce, bit-for-bit, the int16 mono array that madmom's
    ``Signal(path, num_channels=1)`` reads back from a stem wav file written
    by ``demucs_infer.audio.save_audio(wav, path, sr)`` with its defaults
    (``clip='rescale'``, PCM_S 16-bit). Lets the in-memory pipeline skip the
    wav write/read round trip while staying numerically identical to it.

    wav : [channels, samples] float32 tensor (one row of demucs `sources`).

    Verified empirically against the real on-disk round trip (see the perf
    validation script); if demucs-infer ever changes save_audio's defaults
    (clip mode, bit depth) this needs to be re-verified against it.
    """
    wav = prevent_clip(wav, mode='rescale')
    # PCM_S16 quantization -- matches demucs_infer.audio.save_audio's wav16
    # writer exactly (round-half-to-even of x * 2**15, then clamp to int16).
    quantized = (wav.clamp(-1, 1) * 32768).round().clamp(-32768, 32767).to(torch.int16)
    arr = quantized.cpu().numpy().T  # [samples, channels], scipy.io.wavfile's convention
    # madmom's remix() to mono for integer dtypes: mean over the channel axis
    # in float64, then cast back to int16 -- numpy's float->int astype
    # truncates toward zero, it does not round.
    return np.mean(arr, axis=-1).astype(np.int16)


def separate_in_memory(
    paths: List[Path],
    demix_dir: Path,
    device: Union[str, torch.device] = 'auto',
    demucs_overlap: float = 0.25,
    demucs_fp16: bool = False,
    provider: Optional[DemucsProvider] = None,
):
    """Fresh-run fast path for the default DemucsProvider: separates audio
    directly into in-memory mono stem arrays, skipping the stem wav
    write/read round trip (~0.83s/track).

    Tracks whose stems already exist on disk are left untouched and reported
    back as `cached_paths` -- callers must run those through the ordinary
    get_stems()/extract_spectrograms() path so existing keep_byproducts /
    resumed-run caching semantics are preserved exactly.

    Returns
    -------
    cached_paths : List[Path]
        Subset of `paths` whose stems are already cached on disk.
    stems_dirs : Dict[Path, Path]
        For every path in `paths` (cached or fresh): the stems directory that
        holds/would hold its stem wavs, mirroring DemucsProvider.get_stems()'s
        naming, for bookkeeping/cleanup.
    arrays_by_path : Dict[Path, Dict[str, np.ndarray]]
        For the freshly-separated paths only: {'bass': arr, 'drums': arr,
        'other': arr, 'vocals': arr} mono int16 arrays.
    sr_by_path : Dict[Path, int]
        For the freshly-separated paths only: each track's native sample
        rate. Per-track because the disk path preserves each file's own rate
        (_load_input_audio -> save_audio(..., sr) with no resampling), so a
        mixed-rate batch must carry a rate per file, not one shared value.
    """
    if provider is None:
        provider = DemucsProvider(
            device=device,
            demucs_overlap=demucs_overlap,
            demucs_fp16=demucs_fp16,
        )
        # DemucsProvider.__init__ resolves 'auto'/None/etc through the shared
        # resolver -- carry the concrete resolved value forward so the raw
        # (possibly unresolved) `device` local below is never handed to
        # `.to()` directly.
        device = provider.device
    else:
        device = provider.device
        demucs_overlap = provider.demucs_overlap
        demucs_fp16 = provider.demucs_fp16
    required_stems = [f'{name}.wav' for name in STEM_NAMES]

    cached_paths = []
    stems_dirs = {}
    arrays_by_path = {}
    sr_by_path = {}

    for path in paths:
        stems_dir = demix_dir / provider.model_name / Path(path).stem
        stems_dirs[path] = stems_dir
        if all((stems_dir / stem).exists() for stem in required_stems):
            cached_paths.append(path)
            continue

        sources, sr = _run_demucs_separation(
            provider.model, path, device, demucs_overlap, demucs_fp16,
        )

        arrays_by_path[path] = {
            name: quantize_stem_to_madmom_mono_int16(sources[i])
            for i, name in enumerate(provider.model.sources)
        }
        sr_by_path[path] = sr

    return cached_paths, stems_dirs, arrays_by_path, sr_by_path


class PrecomputedStemProvider(StemProvider):
    """Provider for pre-computed stems."""
    
    def __init__(self, stems_mapping: Optional[Dict[str, Path]] = None):
        """
        Initialize with optional stems mapping.
        
        Parameters
        ----------
        stems_mapping : Optional[Dict[str, Path]]
            Mapping from audio identifiers to stem directories
        """
        self.stems_mapping = stems_mapping or {}
    
    def add_stems(self, identifier: str, stems_dir: Path):
        """Add stems for a given identifier."""
        self.stems_mapping[identifier] = Path(stems_dir)
    
    def get_stems(self, identifier: Union[Path, str], output_dir: Path) -> Path:
        """Get pre-computed stems."""
        key = str(identifier)
        if key not in self.stems_mapping:
            raise ValueError(f"No stems found for identifier: {key}")
        
        source_dir = self.stems_mapping[key]
        target_dir = output_dir / Path(identifier).stem
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy or link stems if needed
        required_stems = [f'{name}.wav' for name in STEM_NAMES]
        for stem in required_stems:
            source_stem = source_dir / stem
            target_stem = target_dir / stem
            
            if not target_stem.exists() and source_stem.exists():
                # Create symbolic link to avoid copying large files
                try:
                    target_stem.symlink_to(source_stem.resolve())
                except OSError:
                    # Fallback to copying if symlink fails
                    import shutil
                    shutil.copy2(source_stem, target_stem)
        
        return target_dir


class CustomSeparatorProvider(StemProvider):
    """Provider that uses a custom separation function."""
    
    def __init__(self, separator_fn: StemSeparator):
        """
        Initialize with custom separator function.
        
        Parameters
        ----------
        separator_fn : StemSeparator
            Custom function that implements the StemSeparator protocol
        """
        self.separator_fn = separator_fn
    
    def get_stems(self, identifier: Union[Path, str], output_dir: Path) -> Path:
        """Use custom separator to generate stems."""
        audio_path = Path(identifier)
        # Route through the shared resolver instead of a second, duplicated
        # cuda-if-available check (single owner: `utils.resolve_device`).
        device = resolve_device('auto')
        return self.separator_fn.separate(audio_path, output_dir, device)


def get_stems(
    paths: List[Path],
    stems_dir: Path,
    stem_provider: Optional[StemProvider] = None,
    device: Union[str, torch.device] = 'auto',
    demucs_overlap: float = 0.25,
    demucs_fp16: bool = False,
) -> List[Path]:
    """
    Get stems for audio files using specified provider.

    Parameters
    ----------
    paths : List[Path]
        List of audio file paths
    stems_dir : Path
        Directory to store stems
    stem_provider : Optional[StemProvider]
        Stem provider to use. If None, uses default DemucsProvider
    device : Union[str, torch.device]
        Device to use for separation
    demucs_overlap : float
        EXPERIMENTAL, accuracy-affecting. Only used when stem_provider is None
        (default DemucsProvider). See DemucsProvider.__init__.
    demucs_fp16 : bool
        EXPERIMENTAL, accuracy-affecting. Only used when stem_provider is None
        (default DemucsProvider). See DemucsProvider.__init__.

    Returns
    -------
    List[Path]
        List of paths to directories containing stems
    """
    if stem_provider is None:
        stem_provider = DemucsProvider(device=device, demucs_overlap=demucs_overlap, demucs_fp16=demucs_fp16)
    
    stem_paths = []
    todos = []
    
    for path in paths:
        try:
            stem_path = stem_provider.get_stems(path, stems_dir)
            stem_paths.append(stem_path)
        except Exception as e:
            print(f"Warning: Failed to get stems for {path}: {e}")
            todos.append(path)
    
    if todos:
        print(f"=> Found {len(paths) - len(todos)} tracks with stems ready, {len(todos)} failed.")
        # Could implement fallback logic here if needed
    else:
        print(f"=> All {len(paths)} tracks have stems ready.")
    
    return stem_paths
