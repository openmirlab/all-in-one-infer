#!/bin/bash
# Automated installation script for all-in-one-infer.
# Installs from GitHub source (the maintained channel; PyPI is retired). All
# dependencies, including madmom-infer and demucs-infer, are declared in
# pyproject.toml, so there is no separate dependency step.

set -e

PACKAGE="all-in-one-infer @ git+https://github.com/openmirlab/all-in-one-infer.git"

echo "🚀 Installing all-in-one-infer..."

if command -v uv &> /dev/null; then
    echo "📦 Using uv..."
    uv pip install "$PACKAGE"
elif command -v pip &> /dev/null; then
    echo "📦 Using pip..."
    pip install "$PACKAGE"
else
    echo "❌ Error: Neither uv nor pip found"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ℹ️  FFmpeg not found: WAV/FLAC input works, but MP3 and other formats need ffmpeg on PATH."
fi

echo "✅ Installation complete!"
