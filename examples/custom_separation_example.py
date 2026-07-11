#!/usr/bin/env python3
"""
Example script demonstrating custom source separation integration with All-In-One-Infer.

This script shows different ways to use custom models or pre-computed stems
instead of the default HTDemucs separation.
"""

import json
import torch
import librosa
import soundfile as sf
from pathlib import Path
from allin1_infer import analyze
from allin1_infer.stems import (
    CustomSeparatorProvider,
    PrecomputedStemProvider, 
    DemucsProvider
)


class DummyCustomSeparator:
    """
    Example custom separator that creates dummy stems for demonstration.
    Replace this with your actual model implementation.
    """
    
    def __init__(self, model_name="DummyModel"):
        self.model_name = model_name
        print(f"Initialized {model_name} separator")
    
    def separate(self, audio_path, output_dir, device):
        """
        Separate audio into 4 stems using your custom model.
        
        This is a dummy implementation that creates simple stems for demonstration.
        In a real implementation, you would:
        1. Load your trained model
        2. Process the audio through your model  
        3. Save the separated stems
        """
        print(f"Separating {audio_path.name} using {self.model_name}...")
        
        # Load the original audio
        audio, sr = librosa.load(str(audio_path), sr=44100, mono=False)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Create output directory
        stems_dir = output_dir / 'custom' / audio_path.stem  
        stems_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy separation logic (replace with your model)
        # In a real implementation, this would be something like:
        # stems = your_model.separate(audio)
        
        # Create simple dummy stems for demonstration
        duration = audio.shape[1]
        stems = {}
        
        # Bass: keep low frequencies  
        stems['bass'] = self._filter_audio(audio, sr, low_cut=None, high_cut=250)
        
        # Drums: keep mid frequencies with some percussion emphasis
        stems['drums'] = self._filter_audio(audio, sr, low_cut=100, high_cut=8000) * 0.7
        
        # Vocals: keep high frequencies
        stems['vocals'] = self._filter_audio(audio, sr, low_cut=300, high_cut=None)
        
        # Other: keep everything with reduced amplitude
        stems['other'] = audio * 0.5
        
        # Save stems
        for stem_name, stem_audio in stems.items():
            output_path = stems_dir / f"{stem_name}.wav"
            # Convert to mono if stereo
            if stem_audio.shape[0] > 1:
                stem_audio = librosa.to_mono(stem_audio)
            sf.write(str(output_path), stem_audio, sr)
            print(f"  Saved {stem_name} stem: {output_path}")
        
        print(f"✓ Custom separation complete: {stems_dir}")
        return stems_dir
    
    def _filter_audio(self, audio, sr, low_cut=None, high_cut=None):
        """Apply simple frequency filtering (replace with better filtering in production)."""
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        
        if low_cut is not None and high_cut is not None:
            # Bandpass filter
            low = low_cut / nyquist
            high = high_cut / nyquist
            b, a = butter(5, [low, high], btype='band')
        elif low_cut is not None:
            # High-pass filter  
            low = low_cut / nyquist
            b, a = butter(5, low, btype='high')
        elif high_cut is not None:
            # Low-pass filter
            high = high_cut / nyquist
            b, a = butter(5, high, btype='low')
        else:
            # No filtering
            return audio
        
        # Apply filter to each channel
        import numpy as np
        filtered = []
        for channel in audio:
            filtered.append(filtfilt(b, a, channel))
        
        return np.array(filtered)


def example1_custom_separator():
    """Example 1: Using a custom separator."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Custom Source Separation Model")
    print("="*60)
    
    # Initialize your custom separator
    separator = DummyCustomSeparator("MyAwesomeModel")
    provider = CustomSeparatorProvider(separator)
    
    # Analyze with custom separation
    audio_file = "./assets/Nujabes - Luv(sic) Part 2 feat.Shing02.wav"
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
        return
    
    results = analyze(
        paths=[audio_file],
        stem_provider=provider,
        out_dir="./results_custom",
        demix_dir="./stems_custom",
        keep_byproducts=True  # Keep stems for inspection
    )
    
    print(f"✓ Analysis complete with custom separator!")
    print(f"Results saved to: ./results_custom")
    print(f"Custom stems saved to: ./stems_custom")


def example2_precomputed_stems():
    """Example 2: Using pre-computed stems."""
    print("\n" + "="*60) 
    print("EXAMPLE 2: Pre-computed Stems")
    print("="*60)
    
    # Create a mapping of audio files to existing stem directories
    stems_mapping = {
        "./assets/Nujabes - Luv(sic) Part 2 feat.Shing02.wav": "./stems_custom/custom/Nujabes - Luv(sic) Part 2 feat.Shing02"
    }
    
    # Check if stems exist
    audio_file = list(stems_mapping.keys())[0]
    stems_dir = Path(stems_mapping[audio_file])
    
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        return
        
    if not stems_dir.exists():
        print(f"Stems directory not found: {stems_dir}")
        print("Run example1_custom_separator() first to generate stems")
        return
    
    # Use pre-computed stems
    provider = PrecomputedStemProvider(stems_mapping)
    
    results = analyze(
        paths=[audio_file],
        stem_provider=provider,
        out_dir="./results_precomputed"
    )
    
    print(f"✓ Analysis complete with pre-computed stems!")
    print(f"Results saved to: ./results_precomputed")


def example3_stems_dict_cli():
    """Example 3: Create stems mapping JSON for CLI usage."""
    print("\n" + "="*60)
    print("EXAMPLE 3: CLI with Stems Dictionary") 
    print("="*60)
    
    # Create stems mapping JSON file
    stems_mapping = {
        "./assets/Nujabes - Luv(sic) Part 2 feat.Shing02.wav": "./stems_custom/custom/Nujabes - Luv(sic) Part 2 feat.Shing02"
    }
    
    # Save mapping to JSON
    mapping_file = "./stems_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(stems_mapping, f, indent=2)
    
    print(f"✓ Created stems mapping: {mapping_file}")
    print(f"Now you can use it with CLI:")
    print(f"  all-in-one-infer './assets/Nujabes - Luv(sic) Part 2 feat.Shing02.wav' --stems-dict {mapping_file} -o ./results_cli")


if __name__ == "__main__":
    print("All-In-One-Fix Custom Source Separation Examples")
    print("=" * 60)
    
    # Check if required packages are available
    try:
        import numpy as np
        import scipy.signal
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install: pip install numpy scipy")
        exit(1)
    
    # Run examples
    print("\nRunning examples...")
    
    try:
        example1_custom_separator()
        example2_precomputed_stems() 
        example3_stems_dict_cli()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Examine the generated stems in ./stems_custom/")
        print("2. Compare results in ./results_custom/, ./results_precomputed/, etc.")
        print("3. Try the CLI examples shown above")
        print("4. Integrate your own models following the patterns shown")
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()