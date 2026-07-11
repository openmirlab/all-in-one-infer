# Copyright (c) 2023 Taejun Kim (All-In-One original)
# Copyright (c) 2025 Bo-Yu Chen (Integration modifications)
# SPDX-License-Identifier: MIT

"""
Stem input/output handling for flexible source separation.

This module provides interfaces for working with pre-separated stems,
allowing developers to use custom source separation models or skip
source separation entirely if stems are already available.
"""

import sys
import subprocess
import torch
import torchaudio
from pathlib import Path
from typing import List, Union, Optional, Dict, Callable, Protocol
from abc import ABC, abstractmethod

# Import demucs-infer for source separation
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
from demucs_infer.audio import save_audio, prevent_clip

import numpy as np


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


class DemucsProvider(StemProvider):
    """Default stem provider using integrated separation module with model caching."""

    def __init__(
        self,
        model_name: str = 'htdemucs',
        device: Union[str, torch.device] = 'cuda',
        demucs_overlap: float = 0.25,
        demucs_fp16: bool = False,
    ):
        self.model_name = model_name
        self.device = device
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

    def clear_model_cache(self):
        """Clear cached model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        required_stems = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav']
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

        # Load audio
        if progress_callback:
            progress_callback("Loading audio file", 0.2)

        wav, sr = torchaudio.load(str(audio_path))

        # Ensure stereo (demucs requires 2 channels)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Add batch dimension and move to device
        wav_batch = wav.unsqueeze(0).to(self.device)

        # Apply model
        if progress_callback:
            progress_callback("Separating audio sources", 0.3)

        with torch.no_grad():
            if self.demucs_fp16 and 'cuda' in str(self.device):
                with torch.autocast('cuda', dtype=torch.float16):
                    sources = apply_model(
                        model, wav_batch, device=self.device,
                        progress=bool(progress_callback), overlap=self.demucs_overlap,
                    )
            else:
                sources = apply_model(
                    model, wav_batch, device=self.device,
                    progress=bool(progress_callback), overlap=self.demucs_overlap,
                )

        # Move to CPU immediately to free GPU memory
        sources = sources.cpu()

        # Clean up GPU memory
        del wav_batch
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remove batch dimension
        sources = sources.squeeze(0)

        # Save stems
        if progress_callback:
            progress_callback("Saving separated stems", 0.8)

        source_names = model.sources
        for i, source_name in enumerate(source_names):
            stem_path = stems_dir / f'{source_name}.wav'
            save_audio(sources[i], str(stem_path), sr)

        # Clean up CPU memory
        del wav, sources

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
    # PCM_S16 quantization -- matches torchaudio's soundfile backend exactly
    # (round-half-to-even via torch.round, then clamp to int16 range).
    quantized = (wav.clamp(-1, 1) * 32768).round().clamp(-32768, 32767).to(torch.int16)
    arr = quantized.cpu().numpy().T  # [samples, channels], scipy.io.wavfile's convention
    # madmom's remix() to mono for integer dtypes: mean over the channel axis
    # in float64, then cast back to int16 -- numpy's float->int astype
    # truncates toward zero, it does not round.
    return np.mean(arr, axis=-1).astype(np.int16)


def separate_in_memory(
    paths: List[Path],
    demix_dir: Path,
    device: Union[str, torch.device] = 'cuda',
    demucs_overlap: float = 0.25,
    demucs_fp16: bool = False,
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
        (torchaudio.load -> save_audio(..., sr) with no resampling), so a
        mixed-rate batch must carry a rate per file, not one shared value.
    """
    provider = DemucsProvider(device=device)
    required_stems = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav']

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

        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        wav_batch = wav.unsqueeze(0).to(device)

        with torch.no_grad():
            if demucs_fp16 and 'cuda' in str(device):
                with torch.autocast('cuda', dtype=torch.float16):
                    sources = apply_model(
                        provider.model, wav_batch, device=device,
                        progress=False, overlap=demucs_overlap,
                    )
            else:
                sources = apply_model(
                    provider.model, wav_batch, device=device,
                    progress=False, overlap=demucs_overlap,
                )

        sources = sources.cpu().squeeze(0)
        del wav_batch
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        required_stems = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav']
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.separator_fn.separate(audio_path, output_dir, device)


def get_stems(
    paths: List[Path],
    stems_dir: Path,
    stem_provider: Optional[StemProvider] = None,
    device: Union[str, torch.device] = 'cuda',
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


# Example custom separator implementations
class ExampleCustomSeparator:
    """Example implementation of a custom separator."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Initialize your custom model here
    
    def separate(self, audio_path: Path, output_dir: Path, device: Union[str, torch.device]) -> Path:
        """
        Example custom separation implementation.
        Replace this with your actual separation logic.
        """
        stems_dir = output_dir / 'custom' / audio_path.stem
        stems_dir.mkdir(parents=True, exist_ok=True)
        
        # Your custom separation logic here
        # For example:
        # model = load_your_model(self.model_path, device)
        # stems = model.separate(audio_path)
        # save_stems(stems, stems_dir)
        
        # Placeholder - in real implementation, this would do actual separation
        required_stems = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav']
        for stem in required_stems:
            stem_path = stems_dir / stem
            if not stem_path.exists():
                # Create empty placeholder files - replace with actual separation
                stem_path.touch()
        
        return stems_dir


# Backward compatibility function
def demix(paths: List[Path], demix_dir: Path, device: Union[str, torch.device]) -> List[Path]:
    """
    Legacy demix function for backward compatibility.
    Uses default DemucsProvider.
    """
    return get_stems(paths, demix_dir, None, device)