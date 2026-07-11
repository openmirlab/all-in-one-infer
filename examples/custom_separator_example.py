"""Minimal placeholder StemSeparator implementation.

Shows the shape of the `allin1_infer.stems.StemSeparator` protocol
(`separate(audio_path, output_dir, device) -> Path`) without doing any real
separation -- it just creates empty stem files. Use it as a starting
skeleton for wiring in your own model; for a working example that actually
filters audio, see `custom_separation_example.py` in this directory.
"""

from pathlib import Path
from typing import Union

import torch

from allin1_infer.spectrogram import STEM_NAMES


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
        required_stems = [f'{name}.wav' for name in STEM_NAMES]
        for stem in required_stems:
            stem_path = stems_dir / stem
            if not stem_path.exists():
                # Create empty placeholder files - replace with actual separation
                stem_path.touch()

        return stems_dir
