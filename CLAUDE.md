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
3.0.0 (2026-07) — see [CHANGELOG.md](CHANGELOG.md) for the full
history. This package depends on two sibling packages in the same org for
parts of its pipeline: [demucs-infer](https://github.com/openmirlab/demucs-infer)
(source separation) and [madmom-infer](https://github.com/openmirlab/madmom-infer)
(spectrogram extraction + DBN beat/downbeat decoding) — both are deliberately
delegated to, not vendored, so their own CLAUDE.md/accuracy gates are the
place to look when a bug could be upstream of this package.

The metrical decoder conditionally forwards `fast_viterbi=True` when the
installed `DBNDownBeatTrackingProcessor` signature explicitly supports it.
The legacy constructor remains unchanged for the public `madmom-infer` 0.2.0
compatibility path, so this internal performance adaptation does not require
new package metadata, a public flag, or a decoder thread-count change. It is an
internal newer-revision detail and must not be documented as an unpublished
package release.

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
`utils.resolve_device()` owns strict explicit validation (`auto`, `cpu`,
`cuda`, `cuda:N`) before the Harmonix loader, analysis path, or
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

## Testing philosophy

Plain `pytest tests/ -v` runs all offline contracts, including the committed NATTEN golden
fixture. Eight existing offline modules cover activation metadata, clean API/lifecycle,
checkpoint resolution, device forwarding, and metrical compatibility. NATTEN is not a
dependency, so its committed golden fixture is the only neighborhood-attention reference.

Five model-backed analysis/sonification/visualization tests retain their assertions and are
marked `integration` and `network`. `tests/conftest.py` deselects them by default. Enable
with `--run-integration --integration-audio /path/to/music.wav`; missing input is an error,
not a passing skip. One full CPU Harmonix/Demucs run is shared across these tests, and
byproducts go to pytest temporary directories. No hardcoded local demo is required.

Three historical `*original*.py` files were unasserted diagnostic scripts, including one
that copied audio into an upstream checkout. Their hashes and prior source commit are
recorded in `tests/fixtures/delivery_baseline.json`; `tools/README.md` preserves their intent
and historical observations. `tools/reference_diagnostic.py` replaces them with explicit
interpreter/audio/output arguments, temporary working/cache directories, disabled bytecode
writes, no package installation or input copying, and nonzero failures. Its reference tree
preservation and failure behavior are tested with a separate fake upstream interpreter.

The pre-edit offline baseline was 40 passed / 2 optional live-NATTEN skips on Python 3.11.13,
Torch/Torchaudio 2.7.1, NumPy 2.4.6, SciPy 1.17.1. Runtime source hashes and numerical fixture
identity are committed. That environment used editable madmom-infer 0.3.0 from a sibling;
it does not prove fresh published-dependency resolution. Pre-existing repo-wide ruff debt
was 121 findings (63 runtime, 45 tests, 13 examples); delivery work lints its touched verification
tooling and does not widen into a production lint cleanup.

## NATTEN removed (2026-10)

Neighborhood attention runs only on the pure-PyTorch implementation in
`src/allin1_infer/models/neighborhood_attention.py`, numerically identical to
NATTEN 0.17.5 (`tests/test_neighborhood_attention.py` against the committed
`tests/fixtures/natten_0_17_5_golden.pt`). The former optional `[natten]`
fused-kernel extra and `dinat.py`'s import-time backend selection were removed
because no natten release fits this package:

- `natten` 0.17.x-0.19.x's C++ extension does not compile against torch>=2.8
  (`_device_t` was removed from torch's C++ API), so the extra had to cap
  torch at `<2.8.0`. Checked 2026-09 against the fleet's `torch==2.13.0` pin:
  installing the extra either silently downgraded torch to 2.7.1 or failed
  resolution when torch was pinned exactly.
- natten>=0.20 dropped the functional/RPB API (`na1d_av`/`na1d_qk`/
  `na2d_av`/`na2d_qk` with `rpb`) that the `harmonix-*` checkpoints need.

Do not reintroduce NATTEN as a runtime backend; the golden fixture is the
reference for any change to `neighborhood_attention.py`.

## Verification commands

```bash
# Full offline suite; integration tests are explicitly deselected
python -m pytest tests/ -v

# Optional full-model integration tests with independent audio (may download weights)
python -m pytest tests/ -v --run-integration --integration-audio /path/to/music.wav -m integration

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
  natten 0.21.0+, both wrong today — natten is not used at all (see "NATTEN
  removed" above).
  The root README.md and this file are the accurate, current source; these
  three are left as historical record rather than rewritten, matching the
  `docs/README.md` precedent above. Do not use them as install/compat
  guidance.
- Historical unasserted original-package scripts are preserved by git history and the
  provenance record in `tools/README.md`; use the safe reference diagnostic there.

## Delivery verification

Hatchling reads the version from `src/allin1_infer/__about__.py`; that version file remains
unchanged. The two production-file exceptions are
`checkpoints.py` and `session.py`: each adds only `from __future__ import annotations` to fix
reproduced Python 3.9 `TypeError`s from evaluated union annotations. The initial checkpoint-only
preflight missed the session annotation; the full hosted Python 3.9 suite exposed it. No function body,
model code, numerical dependency, or supported Python floor changed. Python remains `>=3.9`,
with classifiers and the workflow matrix covering 3.9–3.12. The contradictory 3.8 classifier
was removed; no numerical dependency floor or optional NATTEN ceiling changed. `pytest>=8.0`
is declared in the dev extra, allowing a compatible pytest release on Python 3.9.

`.github/workflows/verify.yml` is reusable via `workflow_call` and also runs on PRs/main
pushes. It installs CPU Torch only, then installs `.[dev]` without requesting dependency upgrades,
and asserts that TorchAudio is absent: the package must import and run without it (issue
#7), so the suite runs in exactly that environment. FFmpeg is installed for the lossy-input
decoding tests. Each matrix entry runs the
full offline suite, lints touched delivery tooling, builds a wheel from the sdist, reinstalls
that wheel, and exercises public imports/configuration and all NATTEN golden fields from
outside the checkout. Failed jobs retain dependency/Torch diagnostics and pytest results.
`publish.yml` requires this entire workflow before publishing; its obsolete Torch/setuptools
build dependencies and unquoted shell constraint were removed. Running local verification
does not trigger publication.

The sdist includes tests, golden fixtures, tools, and maintainer docs; the wheel includes
only runtime package files/checkpoint configuration plus distribution metadata.
`tools/installed_smoke.py` verifies untouched installed `.py`/`.toml` bytes against the committed pre-edit hashes.
Approved later rewrites (`stems.py` for issue #7; `models/dinat.py` and
`models/neighborhood_attention.py` for the NATTEN removal) must instead equal the final source bytes.
For `checkpoints.py`, it verifies final-source byte equality, removes exactly the single
approved future import, and then requires the original hash. It also checks all 24 NATTEN
output fields at the original tolerances.
It does not require model downloads or claim full pretrained-pipeline accuracy.

```bash
python -m pip install -e ".[dev]" build
python -m pytest tests/ -v -ra
ruff check tools tests/conftest.py tests/test_reference_diagnostic.py tests/test_checkpoint_config.py tests/test_analyze.py tests/test_sonify.py tests/test_visualize.py
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
# From outside this checkout (substitute its absolute path):
cd /tmp
python /path/to/all-in-one-infer/tools/installed_smoke.py /path/to/all-in-one-infer/tests/fixtures/natten_0_17_5_golden.pt /path/to/all-in-one-infer/tests/fixtures/delivery_baseline.json
all-in-one-infer --help
```

Public skills check (2026-10-02): `openmirlab-skills/plugins/mir/CLAUDE.md` retains the same
`pip install all-in-one-infer`, session, one-shot, and direct-stems guidance. Runtime API and
user installation commands are unchanged, so no public skills edit is required.

## Distribution policy (2026-10-05)

Install the current source from `https://github.com/openmirlab/all-in-one-infer`. GitHub release workflows verify and build distributions but do not upload to PyPI. Existing PyPI versions, where any exist, are historical snapshots. Update installation examples to use Git when changing this package.
