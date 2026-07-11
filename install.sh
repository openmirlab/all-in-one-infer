#!/bin/bash
# Automated installation script for all-in-one-fix
# NATTEN is no longer required — the only extra step is madmom (git-only dependency).

set -e

echo "🚀 Installing all-in-one-fix..."

if command -v uv &> /dev/null; then
    echo "📦 Using uv..."
    uv pip install all-in-one-fix
    uv pip install git+https://github.com/CPJKU/madmom
    echo "✅ Installation complete!"
elif command -v pip &> /dev/null; then
    echo "📦 Using pip..."
    pip install all-in-one-fix
    pip install git+https://github.com/CPJKU/madmom
    echo "✅ Installation complete!"
else
    echo "❌ Error: Neither uv nor pip found"
    exit 1
fi
