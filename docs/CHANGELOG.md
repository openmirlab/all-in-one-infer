# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Validate explicit runtime devices (`cpu`, `cuda`, `cuda:N`, and supported
  `mps`) before forwarding them through Harmonix loading, analysis, and the
  session-owned Demucs provider.
- `AllInOneSession.cache_info()` now applies the same custom checkpoint
  metadata overrides that `load()` forwards to the loader. Python 3.10 uses
  the conditional `tomli` TOML parser fallback.

## [3.1.0] - 2026-07-12

### Changed
- **madmom dependency replaced by [madmom-infer](https://github.com/openmirlab/madmom-infer)**:
  spectrogram extraction (`spectrogram.py`) and DBN downbeat decoding
  (`postprocessing/metrical.py`) now import from `madmom_infer` instead of
  `madmom`. madmom-infer is a from-scratch, pure numpy/scipy reimplementation
  of the madmom surface this project uses, published to PyPI — no more
  `pip install git+https://github.com/CPJKU/madmom` step, no post-install
  hook, no git dependency in package metadata (PyPI rejects those anyway).
  `setup.py` (which existed solely to auto-install madmom from git after
  `pip install`) is removed; the version is now read directly from
  `__about__.py` via `[tool.setuptools.dynamic]`.
- **Verification**: closure proof showed the only numerical difference
  between madmom and madmom-infer is madmom-infer's correctly-rounded STFT
  magnitude (madmom's own `np.abs` has a known 1-ULP rounding bug) — i.e.
  madmom-infer is more accurate, not less. End-to-end on 3 real songs, every
  discrete output (bpm/beats/downbeats/segments) was exactly identical
  between the two backends; raw activations differ by at most 2e-4.
- A fresh-venv install/import/pytest run (no git madmom involved anywhere)
  reproduces the exact same pass/skip test outcomes as the pre-swap baseline.

### Fixed
- Revived three silently-dead test modules (`tests/test_analyze.py`,
  `tests/test_sonify.py`, `tests/test_visualize.py`): they still imported the
  pre-rename `allin1` package (dead since the 3.0.0 rename) and referenced
  never-committed `tests/test.mp3`/`tests/test.json` assets. They now run a
  real end-to-end pipeline against the bundled `assets/` demo track via a
  shared session-scoped `analyze()` fixture (`tests/conftest.py`) — one full
  demucs + harmonix-all ensemble run per pytest session, byproducts written
  to a pytest tmp dir. The demo track is a .wav on purpose: sonify's output
  file mirrors the source suffix, and .mp3 output would require the optional
  `demucs-infer[mp3]` (lameenc) extra.
- **Fresh installs on torchaudio>=2.11 crashed loading the user's input
  audio.** `stems.py`'s `_run_demucs_separation` had its own unguarded
  `torchaudio.load(str(audio_path))` call (for the ORIGINAL input file the
  user passed in — not demucs's own stem output) alongside demucs-infer's,
  raising `ImportError('TorchCodec is required for load_with_torchcodec...')`
  once torchaudio dropped its bundled decoders. Reproduced end-to-end with a
  fresh `uv venv` letting the resolver pick torch 2.13.0 / torchaudio 2.11.0:
  13 passed / 2 skipped / **5 errors** before the fix, 18 passed / 2 skipped
  / 0 errors after — matching the pre-existing swap-branch baseline exactly.
  Fixed the same way demucs-infer 4.2.2 fixed its own equivalent call: wav/
  flac now load via `soundfile` (verified bit-identical to `torchaudio.load`
  — `np.array_equal` exact — on PCM16/24/32 wav, FLAC, mono/stereo, and the
  two real multi-minute assets under `assets/`). mp3 stays `torchaudio`-only
  — the same check measured mp3 decode to differ by up to `2.4e-6` per
  sample between `torchaudio` and `soundfile` (consistent with
  demucs-infer's own `~7e-7` finding), so it's never silently routed through
  a different decoder — a clear, actionable error (install `torchcodec`, or
  convert to wav/flac) is raised instead if `torchaudio` itself can't decode.
  mp3 is a documented, first-class input format here (see "Concerning MP3
  Files" below), unlike demucs-infer's own call site which only ever reads
  its own wav/flac stem output — so this fix's format split had to be
  derived independently rather than copied verbatim.
- Bumped the `demucs-infer` dependency floor to `>=4.2.2` for its own
  torchaudio>=2.11/torchcodec fix (and the earlier bit-exact-fixture/
  natten-era fixes); added `soundfile>=0.12.1` as an explicit core
  dependency (previously only pulled in transitively).

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

### Added
- Added `AllInOneSession`, an independent reusable lifecycle facade with
  `load`, ready-only `infer`, `release`, `close`, `status`, `cache_info`, and
  context-manager support. Existing lazy `analyze()` calls remain compatible.
- Added package-owned `config/checkpoints.toml` and validated metadata helpers
  with generic config/path/URL/checksum override seams.

### Changed
- `AllInOneSession` now owns and reuses the complete default mixed-input
  pipeline lifecycle: `load()` prepares Harmonix plus one lazy HTDemucs
  provider, `infer()` injects that provider into the verified
  in-memory separation path and the keep-byproducts disk path, `release()` /
  `close()` free both components, and `cache_info()` reports both lifecycle
  components. Direct-stems inference never loads Demucs, while legacy lazy
  `analyze()` behavior and the direct-stems API remain unchanged.

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

## [2.0.0] - historical

First release of this fork under the (now-retired) `all-in-one-fix` /
`allin1fix` naming, before the 3.0.0 rename to `all-in-one-infer` /
`allin1_infer`. Recorded here for history; see [3.0.0](#300---2026-07-11)
above for the rename itself.

### Changed

- **NATTEN dependency removed as a hard requirement**: reimplemented NATTEN's
  neighborhood attention (1D + 2D, with relative positional biases) in pure
  PyTorch (`src/allin1_infer/models/neighborhood_attention.py`), numerically
  identical to NATTEN 0.17.5 per golden-fixture tests (forward and backward).
  Pretrained checkpoints load unchanged. Installs anywhere with a single
  `pip install` -- any torch >= 2.0, CPU-only machines, and platforms NATTEN
  never supported (e.g. macOS / Apple Silicon). If a compatible NATTEN
  (0.17.x-0.19.x) is installed, it's used automatically as a faster
  fused-kernel backend.
- **Source separation switched to demucs-infer**: replaced the original,
  unmaintained `demucs` dependency (PyTorch 1.x only) with `demucs-infer`
  (PyTorch 2.x compatible, actively maintained). Added intelligent model
  caching (~6x faster on repeated use) and automatic GPU memory cleanup.
- **Modern packaging**: converted to `pyproject.toml` (PEP 621) with
  hatchling as the build backend; full pip and UV compatibility.

### Added

- Cache management: `--cache-info` / `--clear-cache` CLI flags and
  `print_cache_info()` / `clear_model_cache()` Python API.
- Flexible stems input: custom separation models via a pluggable provider
  interface, pre-computed stems from any tool, direct stems input (skip
  separation entirely), and hybrid workflows mixing all three.
- Fuzzy-matching error messages for invalid model names.

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
