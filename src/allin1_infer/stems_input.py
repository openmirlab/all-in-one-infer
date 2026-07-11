"""
Direct stems input functionality for All-In-One.

This module enables users to directly provide separated stems (bass, drums, other, vocals)
as input instead of a single audio track, completely bypassing source separation.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from .typings import PathLike


@dataclass
class StemsInput:
    """Container for direct stems input."""
    bass: Path
    drums: Path
    other: Path
    vocals: Path
    identifier: Optional[str] = None  # Optional identifier for the stem set
    
    def __post_init__(self):
        """Validate that all stem files exist."""
        stems = {'bass': self.bass, 'drums': self.drums, 'other': self.other, 'vocals': self.vocals}
        
        for stem_name, stem_path in stems.items():
            if not Path(stem_path).exists():
                raise FileNotFoundError(f"Stem file not found: {stem_name} at {stem_path}")
    
    @property
    def name(self) -> str:
        """Generate a name for this stem set."""
        if self.identifier:
            return self.identifier
        
        # Use bass filename as identifier (remove extension)
        return Path(self.bass).stem
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            'bass': str(self.bass),
            'drums': str(self.drums),
            'other': str(self.other),
            'vocals': str(self.vocals),
            'identifier': self.identifier or self.name
        }


def validate_stems_input(stems_input: Union[StemsInput, Dict[str, PathLike]]) -> StemsInput:
    """
    Validate and normalize stems input.
    
    Parameters
    ----------
    stems_input : Union[StemsInput, Dict[str, PathLike]]
        Either a StemsInput object or dict with keys: bass, drums, other, vocals
        
    Returns
    -------
    StemsInput
        Validated stems input object
    """
    if isinstance(stems_input, dict):
        required_keys = {'bass', 'drums', 'other', 'vocals'}
        provided_keys = set(stems_input.keys())
        
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(f"Missing required stem keys: {missing}")
        
        stems_input = StemsInput(
            bass=Path(stems_input['bass']),
            drums=Path(stems_input['drums']),
            other=Path(stems_input['other']),
            vocals=Path(stems_input['vocals']),
            identifier=stems_input.get('identifier')
        )
    
    if not isinstance(stems_input, StemsInput):
        raise TypeError(f"Expected StemsInput or dict, got {type(stems_input)}")
    
    return stems_input


def copy_stems_to_demix_structure(stems_input: StemsInput, demix_dir: Path) -> Path:
    """
    Copy stem files to the expected demix directory structure.
    
    Parameters
    ----------
    stems_input : StemsInput
        Validated stems input
    demix_dir : Path
        Base demix directory
        
    Returns
    -------
    Path
        Path to the created stems directory
    """
    # Create target directory in standard demix structure
    target_dir = demix_dir / 'stems_input' / stems_input.name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy each stem file
    stems_map = {
        'bass.wav': stems_input.bass,
        'drums.wav': stems_input.drums, 
        'other.wav': stems_input.other,
        'vocals.wav': stems_input.vocals
    }
    
    for target_name, source_path in stems_map.items():
        target_path = target_dir / target_name
        
        # Only copy if target doesn't exist or is different
        if not target_path.exists() or not files_are_same(source_path, target_path):
            shutil.copy2(source_path, target_path)
    
    return target_dir


def link_stems_to_demix_structure(stems_input: StemsInput, demix_dir: Path) -> Path:
    """
    Create symbolic links to stem files in the expected demix directory structure.
    This avoids copying large audio files.
    
    Parameters
    ----------
    stems_input : StemsInput
        Validated stems input
    demix_dir : Path
        Base demix directory
        
    Returns
    -------
    Path
        Path to the created stems directory
    """
    # Create target directory
    target_dir = demix_dir / 'stems_input' / stems_input.name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for each stem
    stems_map = {
        'bass.wav': stems_input.bass,
        'drums.wav': stems_input.drums,
        'other.wav': stems_input.other, 
        'vocals.wav': stems_input.vocals
    }
    
    for target_name, source_path in stems_map.items():
        target_path = target_dir / target_name
        
        # Remove existing link/file if it exists
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        
        try:
            # Create symbolic link
            target_path.symlink_to(Path(source_path).resolve())
        except OSError:
            # Fallback to copy if symlink fails (e.g., on Windows without permissions)
            shutil.copy2(source_path, target_path)
    
    return target_dir


def files_are_same(file1: Path, file2: Path) -> bool:
    """Check if two files are the same (size and modification time)."""
    try:
        stat1 = Path(file1).stat()
        stat2 = Path(file2).stat()
        return (stat1.st_size == stat2.st_size and 
                stat1.st_mtime == stat2.st_mtime)
    except (OSError, FileNotFoundError):
        return False


def prepare_stems_for_analysis(
    stems_inputs: List[Union[StemsInput, Dict[str, PathLike]]], 
    demix_dir: Path,
    use_symlinks: bool = True
) -> List[Path]:
    """
    Prepare multiple stems inputs for analysis.
    
    Parameters
    ----------
    stems_inputs : List[Union[StemsInput, Dict[str, PathLike]]]
        List of stems inputs to prepare
    demix_dir : Path
        Base demix directory
    use_symlinks : bool
        Whether to use symbolic links (faster) or copy files
        
    Returns
    -------
    List[Path]
        List of paths to prepared stem directories
    """
    prepared_paths = []
    
    for stems_input in stems_inputs:
        # Validate input
        validated_stems = validate_stems_input(stems_input)
        
        # Prepare stems directory
        if use_symlinks:
            stems_dir = link_stems_to_demix_structure(validated_stems, demix_dir)
        else:
            stems_dir = copy_stems_to_demix_structure(validated_stems, demix_dir)
        
        prepared_paths.append(stems_dir)
        print(f"✓ Prepared stems: {stems_dir}")
    
    return prepared_paths


# Convenience functions for common use cases

def create_stems_input_from_directory(stems_dir: PathLike, identifier: Optional[str] = None) -> StemsInput:
    """
    Create StemsInput from a directory containing the 4 stem files.
    
    Parameters
    ----------
    stems_dir : PathLike
        Directory containing bass.wav, drums.wav, other.wav, vocals.wav
    identifier : Optional[str]
        Custom identifier for the stem set
        
    Returns
    -------
    StemsInput
        Validated stems input object
    """
    stems_dir = Path(stems_dir)
    
    if not stems_dir.is_dir():
        raise ValueError(f"Stems directory not found: {stems_dir}")
    
    return StemsInput(
        bass=stems_dir / 'bass.wav',
        drums=stems_dir / 'drums.wav',
        other=stems_dir / 'other.wav',
        vocals=stems_dir / 'vocals.wav',
        identifier=identifier or stems_dir.name
    )


def create_stems_input_from_pattern(
    base_path: PathLike, 
    pattern: str = "{stem}.wav",
    identifier: Optional[str] = None
) -> StemsInput:
    """
    Create StemsInput using a filename pattern.
    
    Parameters
    ----------
    base_path : PathLike
        Base directory or path prefix
    pattern : str
        Pattern for stem files. Use {stem} placeholder for bass/drums/other/vocals
    identifier : Optional[str]  
        Custom identifier for the stem set
        
    Returns
    -------
    StemsInput
        Validated stems input object
        
    Examples
    --------
    >>> # Pattern: song_bass.wav, song_drums.wav, etc.
    >>> stems = create_stems_input_from_pattern("./audio", "song_{stem}.wav")
    
    >>> # Pattern: bass_track.wav, drums_track.wav, etc.  
    >>> stems = create_stems_input_from_pattern("./stems", "{stem}_track.wav")
    """
    base_path = Path(base_path)
    
    # If base_path is a directory, construct full paths
    if base_path.is_dir():
        stem_paths = {
            stem: base_path / pattern.format(stem=stem)
            for stem in ['bass', 'drums', 'other', 'vocals']
        }
    else:
        # If base_path is a file pattern, use its parent directory
        parent_dir = base_path.parent
        stem_paths = {
            stem: parent_dir / pattern.format(stem=stem)
            for stem in ['bass', 'drums', 'other', 'vocals']
        }
    
    return StemsInput(
        bass=stem_paths['bass'],
        drums=stem_paths['drums'],
        other=stem_paths['other'],
        vocals=stem_paths['vocals'],
        identifier=identifier or f"{base_path.stem}_stems"
    )