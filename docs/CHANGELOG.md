# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
