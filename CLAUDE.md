# all-in-one-infer

**`docs/`** is local-only by policy (2026-09-14): kept on disk, gitignored,
never pushed to GitHub. Links below to `docs/*.md` resolve locally only.

Inference-only fork of [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)
(music structure analysis: tempo, beats, downbeats, functional segments).
Training code has been removed; this package only loads pretrained
`harmonix-*` checkpoints and runs inference. See README.md's
["Why This Exists"](README.md#why-this-exists) and
["Scope"](README.md#scope) for the full rationale — this file covers
conventions and verification, not the "why".

## Status

Actively maintained, published to PyPI as `all-in-one-infer`
(`Development Status :: 5 - Production/Stable` in `pyproject.toml`; version
is single-sourced from `src/allin1_infer/__about__.py`). Renamed from
`all-in-one-fix` / `allin1fix` to `all-in-one-infer` / `allin1_infer` as of
3.0.0 (2026-07) — see [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full
history. This package depends on two sibling packages in the same org for
parts of its pipeline: [demucs-infer](https://github.com/openmirlab/demucs-infer)
(source separation) and [madmom-infer](https://github.com/openmirlab/madmom-infer)
(spectrogram extraction + DBN beat/downbeat decoding) — both are deliberately
delegated to, not vendored, so their own CLAUDE.md/accuracy gates are the
place to look when a bug could be upstream of this package.

## Testing philosophy

## Clean API and lifecycle contract

`AllInOneSession` is the explicit lifecycle facade for reusable inference:
call `load()` before ready-only `infer()`, then `release()` or `close()` to
free model resources. For mixed-input runs it owns and reuses both parts of
the default pipeline: the eagerly loaded Harmonix structure model and one
session-owned HTDemucs separator that loads on first mixed inference;
direct-stems input does not load Demucs, and the legacy
`analyze()` function remains the lazy, backward-compatible one-shot path.
`config/checkpoints.toml` owns checkpoint URLs and provenance; callers may
override its path and metadata generically.
`utils.resolve_device()` owns strict explicit validation (`cpu`, `cuda`,
`cuda:N`, plus supported `mps`) before the Harmonix loader, analysis path, or
session-owned Demucs provider receives a device. `stems.py`'s legacy direct
entry points (`DemucsProvider`, `separate_in_memory()`, the module-level
`get_stems()`, `CustomSeparatorProvider`) route through the same resolver
too (2026-09, phonon-readiness fix) — they previously hardcoded a literal
`device='cuda'` default (or, for `CustomSeparatorProvider`, a second inline
cuda-if-available check) that bypassed `resolve_device()` entirely, so
calling them directly without an explicit device never validated an
explicit-but-unavailable request and had no `'auto'` support. `analyze()`
and `AllInOneSession` were never affected — both already resolved `device`
once at their own entry point and threaded the concrete resolved string down
into `stems.py`.

`pyproject.toml` declares no pytest markers or `addopts` — the whole
`tests/` directory runs by default with a plain `pytest` invocation. There
is no built-in `network`/`not network` split like demucs-infer has, even
though several tests do hit the network (checkpoint downloads on first use)
and read fixed audio assets under `assets/`.

Test suite composition (`tests/`), as of this writing:

- **End-to-end pipeline tests** (`test_analyze.py`, `test_sonify.py`,
  `test_visualize.py`): share one session-scoped `analyze()` fixture
  (`tests/conftest.py`) that runs a full demucs + harmonix-all ensemble
  pass on a bundled demo track once per session, then all three modules
  assert against that shared result. Byproducts go to a pytest tmp dir,
  never into the repo. These only run from a source checkout (the demo
  track isn't packaged into the wheel) and need network on first run to
  populate the model cache.
- **Golden-fixture correctness test** (`test_neighborhood_attention.py`):
  the correctness contract for the pure-PyTorch neighborhood-attention
  reimplementation — verifies it's numerically identical to real NATTEN
  0.17.x output. Treat this test as load-bearing: any change to
  `src/allin1_infer/models/neighborhood_attention.py` must keep it passing.
- **Activation/metadata tests** (`test_activation_fps.py`): fast, no
  network or model weights required.
- **`test_original_allinone.py` / `test_original_comparison.py`**:
  pre-rename, exploratory debugging scripts (they `import allinone` — the
  *original* upstream package, not this one — and reference hardcoded local
  asset paths that no longer match this repo's `assets/` layout). They are
  collected by pytest but are not reliable CI tests; treat failures here as
  expected/ignorable rather than a regression signal until someone
  deliberately rewrites or removes them.

## The `[natten]` extra's torch ceiling (justified, checked 2026-09)

`pyproject.toml`'s `[natten]` extra pins `natten>=0.17.1,<0.20` and
`torch>=2.0.0,<2.8.0`. This is a real, confirmed incompatibility, not an
unjustified ceiling (org constitution art. 3 requires evidence for any
upper bound):

- `natten` 0.17.x-0.19.x's C++ extension does not compile against
  torch>=2.8 (`_device_t` was removed from torch's C++ API); natten's own
  install docs confirm this generation is the one this repo needs
  (`na1d_av`/`na1d_qk`/`na2d_av`/`na2d_qk` in `src/allin1_infer/models/dinat.py`).
  natten>=0.20 dropped that functional/RPB API entirely, so it can't run
  the `harmonix-*` checkpoints even though it does support newer torch.
  There is currently no natten release that satisfies both constraints at
  once.
- Verified 2026-09 against the fleet's `torch==2.13.0` pin: `uv pip install
  -e ".[natten]"` (no exact torch pin) resolves by silently **downgrading**
  torch to 2.7.1 (+ matching torchaudio/triton) to satisfy the extra's own
  ceiling — worth knowing if anyone ever installs this extra into a shared
  venv. `uv pip install -e ".[natten]" "torch==2.13.0"` (the fleet's actual
  shape: torch pinned exactly) instead fails **loudly** with a clear
  unsatisfiable-dependencies error, which is the correct/safe outcome.
- **Verdict: keep the ceiling.** The core package (no extras) has no torch
  ceiling and installs/imports cleanly against torch 2.13.0 — confirmed by
  `uv pip install -e .` plus an import smoke test in an isolated venv. Any
  phonon provider on the shared torch-2.13.0 venv must not install the
  `[natten]` extra; the pure-PyTorch neighborhood-attention backend (the
  default) is what actually runs there, and it's numerically identical to
  NATTEN's output (golden-fixture tested, `tests/test_neighborhood_attention.py`).

## Verification commands

```bash
# Full suite (no marker filtering exists yet -- expect network calls on
# first run to populate the model cache, and the two "original_*" scripts
# above to be unreliable)
uv run pytest tests/ -v

# Just the load-bearing correctness gate for neighborhood attention
uv run pytest tests/test_neighborhood_attention.py -v

# Just the fast, no-network activation/metadata tests
uv run pytest tests/test_activation_fps.py -v

# Focused lifecycle/parity checks for session-owned Harmonix + Demucs reuse
uv run pytest tests/test_clean_api.py -v

# Verify install + CLI wiring after any packaging change
python -c "import allin1_infer; print(allin1_infer.__version__)"
all-in-one-infer --help
```

## File-top header convention

Load-bearing files carry a file-top docstring header, matching the sibling
`demucs-infer` package's convention:

```python
"""<Title -- one line>

<2-3 sentences: what this file does, key design decisions, gotchas a
reader would otherwise discover the hard way.>

Reads: <other allin1_infer modules this file imports and why>
"""
```

Example in this repo: `src/allin1_infer/models/neighborhood_attention.py`
explains *why* it exists (NATTEN's compiled-extension and API-removal
problems) and states its correctness contract (numerically identical to
NATTEN 0.17.x, gated by `tests/test_neighborhood_attention.py`) right in
the header, not just in a docstring for its own sake.

## Known, deliberately unfixed issues

- `docs/README.md` (the docs-directory index) still says "All-In-One-Fix"
  in its title, a leftover from before the 3.0.0 rename. Not fixed here
  because it's a low-traffic internal index, not user-facing like the root
  README.md; fix opportunistically alongside other docs/ edits rather than
  as a standalone change.
- `docs/RELEASE_SUMMARY.md`, `docs/PACKAGE_STRUCTURE.md`, and
  `docs/PYPI_PUBLISHING.md` are stale v2.0.0-era planning notes from before
  the 3.0.0 pure-PyTorch neighborhood-attention rewrite (checked 2026-09,
  phonon-readiness pass): they describe `natten` as a **required** core
  dependency (`dependencies = ["natten==0.17.5"]` /
  `natten>=0.17.5` "flexible: 0.17.5-0.21.0+") and claim compatibility up to
  natten 0.21.0+, both wrong today — natten is an optional `[natten]` extra
  pinned to `>=0.17.1,<0.20` (`pyproject.toml`), because natten>=0.20
  dropped the legacy functional/RPB API this port's `dinat.py` depends on.
  The root README.md and this file are the accurate, current source; these
  three are left as historical record rather than rewritten, matching the
  `docs/README.md` precedent above. Do not use them as install/compat
  guidance.
- `tests/test_original_allinone.py` and `tests/test_original_comparison.py`
  are pre-rename debugging scripts, not maintained regression tests (see
  Testing philosophy above). Not removed here because deciding whether to
  delete vs. rewrite them as real regression tests is a judgment call
  outside a docs-conformance pass.
