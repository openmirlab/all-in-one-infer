# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Training code (`src/allin1fix/training/`), the `allin1fix-train` and
  `allin1fix-preprocess` CLI commands, and the `train` optional-dependency
  extra (lightning, timm, wandb, mir_eval). This package is now
  inference-only; see [TRAINING.md](TRAINING.md) for pointers to the
  upstream training pipeline.

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
