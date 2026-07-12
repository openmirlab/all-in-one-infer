# all-in-one-infer

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
- `tests/test_original_allinone.py` and `tests/test_original_comparison.py`
  are pre-rename debugging scripts, not maintained regression tests (see
  Testing philosophy above). Not removed here because deciding whether to
  delete vs. rewrite them as real regression tests is a judgment call
  outside a docs-conformance pass.
