# All-In-One-Fix Package Structure

This document describes the package structure and publishing configuration following the pattern from `demucs-infer`.

## Package Information

- **Name**: `allin1fix`
- **Version**: 2.0.0 (from `src/allin1fix/__about__.py`)
- **License**: MIT
- **Build System**: hatchling
- **Python**: 3.10

## Publishing Configuration

### Build System
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Package Metadata

Following the demucs-infer pattern:
- ✅ Structured license field: `{text = "MIT"}`
- ✅ Maintainers field added
- ✅ Enhanced classifiers for PyPI
- ✅ Comprehensive project URLs
- ✅ Both wheel and sdist targets configured

### Dependencies

**Core dependencies from GitHub:**
- `demucs-infer @ git+https://github.com/openmirlab/demucs-infer.git`
- `madmom @ git+https://github.com/CPJKU/madmom`

**Core dependencies from PyPI:**
- `natten>=0.17.5` (flexible: 0.17.5-0.21.0+ for PyTorch 2.0-2.7.0)
- `numpy`, `librosa`, `hydra-core`, `omegaconf`, `huggingface_hub`, `matplotlib`

**Optional dependencies:**
- `natten`: Optional fused GPU kernels for neighborhood attention (speed-only; pure-PyTorch implementation is used otherwise)
- `dev`: For development (black, ruff)

> Note: this package is inference-only. A `train` extra and
> `allin1fix-train`/`allin1fix-preprocess` CLI commands used to exist for
> reproducing the paper's training pipeline; they have been removed. See
> [TRAINING.md](TRAINING.md) for historical reference and pointers to the
> upstream training pipeline.

### Project URLs

```toml
[project.urls]
Homepage = "https://github.com/ChenPaulYu/all-in-one-fix"
Repository = "https://github.com/ChenPaulYu/all-in-one-fix"
Documentation = "https://github.com/ChenPaulYu/all-in-one-fix#readme"
"Bug Reports" = "https://github.com/ChenPaulYu/all-in-one-fix/issues"
"Original All-In-One" = "https://github.com/mir-aidj/all-in-one"
Changelog = "https://github.com/ChenPaulYu/all-in-one-fix/blob/main/docs/CHANGELOG.md"
```

## Installation Methods

### Option 1: UV (Recommended)
```bash
# Install from GitHub
uv pip install git+https://github.com/openmirlab/all-in-one-fix.git

# Editable install for development
git clone https://github.com/openmirlab/all-in-one-fix.git
cd all-in-one-fix
uv pip install -e .
```

### Option 2: pip (Traditional)
```bash
# Install from GitHub
pip install git+https://github.com/openmirlab/all-in-one-fix.git

# Editable install for development
git clone https://github.com/openmirlab/all-in-one-fix.git
cd all-in-one-fix
pip install -e .
```

## CLI Commands

One entry point is configured:

1. `allin1fix` - Main CLI for music structure analysis

## Package Structure

```
all-in-one-fix/
├── pyproject.toml          # Modern packaging configuration
├── README.md               # Comprehensive documentation
├── LICENSE                 # MIT license
├── NOTICE                  # Attribution notices
├── src/
│   └── allin1fix/         # Main package
│       ├── __about__.py   # Version info (2.0.0)
│       ├── __init__.py    # Package exports
│       ├── cli.py         # CLI entry point
│       ├── analyze.py     # Core analysis functions
│       ├── models/        # Model definitions
│       └── postprocessing/# Post-processing
├── docs/                   # Documentation
├── tests/                  # Test suite
└── examples/              # Usage examples
```

## Key Improvements Over Original All-In-One

1. **Modern Packaging**
   - Uses hatchling build backend
   - Follows PEP 621 standards
   - GitHub dependencies properly configured
   - Both UV and pip compatible

2. **Enhanced Metadata**
   - Comprehensive classifiers for PyPI discovery
   - Multiple project URLs for better navigation
   - Maintainer information
   - License file auto-inclusion

3. **Dependency Management**
   - demucs-infer from GitHub (PyTorch 2.x compatible)
   - madmom from GitHub (latest version)
   - NATTEN 0.17.5-0.21.0+ flexible support (automatic version detection)
   - PyTorch 2.0-2.7.0 compatibility
   - CUDA 11.7-12.8 support
   - Clean separation of core vs optional dependencies

4. **Development Tools**
   - Optional dev dependencies (black, ruff)
   - Editable install support

5. **NATTEN Compatibility Layer**
   - Automatic detection of NATTEN version (0.17.5-0.21.0+)
   - Three-tier compatibility system:
     - NATTEN <0.19: Short names (`na1d_av`, `na2d_av`)
     - NATTEN 0.19: Long names (`natten1dav`, `natten2dav`)
     - NATTEN >=0.20: Modern generic API with wrappers

## Comparison with demucs-infer Pattern

| Feature | demucs-infer | allin1fix |
|---------|--------------|-----------|
| Build backend | hatchling | hatchling ✅ |
| License format | `{text = "MIT"}` | `{text = "MIT"}` ✅ |
| Maintainers | Package Maintainers | Package Maintainers ✅ |
| Enhanced classifiers | Yes | Yes ✅ |
| Project URLs | 6 URLs | 6 URLs ✅ |
| Wheel + sdist | Yes | Yes ✅ |
| GitHub deps | N/A | Yes (demucs-infer, madmom) ✅ |
| Optional deps | mp3, quantized | natten, dev ✅ |
| CLI scripts | 1 | 1 ✅ |
| UV compatible | Yes | Yes ✅ |

## Publishing Readiness

The package is now ready for:

1. ✅ **GitHub releases** - Can be installed via `pip install git+...`
2. ✅ **Local installation** - `pip install -e .` for development
3. ✅ **PyPI publishing** - Metadata follows PyPI requirements
4. ✅ **UV compatibility** - Fast installation and resolution
5. ✅ **Dependency resolution** - GitHub dependencies properly configured

## Notes

- **PyTorch requirement**: Users must install PyTorch manually before installing allin1fix (same as demucs-infer)
- **License compatibility**: MIT license compatible with both All-In-One and Demucs
- **Backward compatibility**: All original functionality preserved
- **Attribution**: Proper credit to All-In-One and Demucs in README and NOTICE
