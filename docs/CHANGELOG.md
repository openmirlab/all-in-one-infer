# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] - 2026-07-11

### Added
- `AnalysisResult.activation_fps`: the frame rate (Hz) of the `activations`
  array, populated whenever activations are requested — downstream consumers
  can now map activation frame indices to timestamps (#2). Saved-JSON
  round-trip is backward compatible (older results load with `None`).

### Documentation
- README fully synced with the 3.0.0 reality: official rename announcement
  (all-in-one-fix → all-in-one-infer), stale `allin1` imports in current-usage
  examples corrected to `allin1_infer`, the pure-PyTorch neighborhood-attention
  story replaces obsolete NATTEN-required claims, and the opt-in performance
  flags are documented in both CLI and Python API sections.

## [3.0.0] - 2026-07-11

First release under the new name **all-in-one-infer** (PyPI project renamed
from `all-in-one-fix`; Python package renamed `allin1fix` -> `allin1_infer` —
a breaking import change, hence the major version).

### Changed
- **NATTEN dependency removed**: neighborhood attention is now a pure-PyTorch
  implementation, numerically verified bit-identical to NATTEN 0.17.5 (golden
  fixtures + end-to-end A/B on real music). Installs anywhere — any
  torch >= 2.0 (upper bound lifted), CPU-only, macOS/Apple Silicon, Windows,
  no C++ toolchain. NATTEN remains available as an optional [natten] extra.
- ~25% faster inference with bit-identical output: TF32 matmul, parallel
  ensemble checkpoint downloads, in-memory stem pipeline (skips the stem
  wav write/read round trip).
- Deep-module refactor: canonical STEM_NAMES, deduplicated Demucs run logic,
  analyze() path selection extracted, helpers/cache split, dead code removed.
- File-top navigation headers across the package; version single-sourced.

### Added
- Opt-in performance knobs (default off, accuracy-affecting ones documented):
  compile_model, demucs_overlap, demucs_fp16.
- Golden-fixture test suite for the neighborhood-attention implementation;
  publish workflow now gates on tests.

### Removed
- Training code (the package is inference-only; see upstream
  mir-aidj/all-in-one for the training pipeline).

## [Unreleased]

### Changed

- Project renamed: PyPI package `all-in-one-fix` → `all-in-one-infer`,
  Python import package `allin1fix` → `allin1_infer` (now at
  `src/allin1_infer/`), and CLI command `allin1fix` → `all-in-one-infer`
  (matching the sibling `demucs-infer` package's convention of using its
  full CLI name rather than an abbreviation). The GitHub repository itself
  is being renamed separately from `all-in-one-fix` to `all-in-one-infer`.

### Removed

- Training code (`src/allin1_infer/training/`), the `allin1_infer-train` and
  `allin1_infer-preprocess` CLI commands, and the `train` optional-dependency
  extra (lightning, timm, wandb, mir_eval). This package is now
  inference-only; see [TRAINING.md](TRAINING.md) for pointers to the
  upstream training pipeline.

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
