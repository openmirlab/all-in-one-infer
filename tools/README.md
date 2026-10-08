# Verification and reference diagnostics

The default `pytest tests/ -v` suite runs the offline contracts and committed NATTEN
fixtures. Model-backed tests are explicitly enabled with:

```bash
pytest tests/ -v --run-integration --integration-audio /absolute/path/to/music.wav -m integration
```

The WAV must be provided independently. Missing input fails before execution. This run may
download Harmonix/Demucs checkpoints; inference byproducts use pytest temporary directories.
The five analysis/sonification/visualization tests retain their assertions.

## Reference diagnostic

Use an already provisioned upstream interpreter and explicit audio/report paths:

```bash
python tools/reference_diagnostic.py \
  --reference-python /absolute/path/to/upstream-env/bin/python \
  --module allin1 \
  --output /absolute/path/to/new-report.json \
  /absolute/path/to/music.wav
```

The report path must be new and outside the reference checkout. The tool installs nothing,
reads audio in place, invokes the reference interpreter from a temporary directory with
bytecode writing disabled, and redirects model caches and analysis byproducts there. It
fails on missing input/interpreter, import/inference errors, or empty BPM/beat results.
Reports describe reference observations; they do not establish numerical parity or model
quality. Temporary caches are discarded after the run, so model downloads may recur.

## Historical diagnostics and baseline provenance

At source commit `5589ea9d0baa1e43c82ab43047c63ade48db14a8`, these three files were collected
as pytest tests despite having no assertions:

- `tests/test_allin1_original.py`: upstream import/audio inspection and a cwd text report.
- `tests/test_original_allinone.py`: hardcoded-track inspection, exception printing, and an
  automatic package-install path when invoked as a script.
- `tests/test_original_comparison.py`: copied tracks into a sibling upstream checkout and
  printed comparison summaries. Its `bmp` typo could interrupt the summary.

The safe CLI above replaces their diagnostic purpose. Original files remain recoverable
from that commit; their SHA-256 values are preserved in
`tests/fixtures/delivery_baseline.json`. No upstream checkout was modified or original
script executed during the delivery change.

The old comparison script printed these hardcoded "our results" values:

| Track label | BPM | Beats | Maximum activation | Claimed success |
|---|---:|---:|---:|---|
| Sunflower 60BPM | 60 | 272 | 1.0 | true |
| Nujabes — Luv(sic) Part 2 | absent | 0 | 1.0 | false |
| NewJeans — Super Shy | absent | 0 | 0.114 | false |

These are unverified historical constants, not recorded golden outputs or current accuracy
claims. The useful current numerical evidence is the committed NATTEN fixture, which is
checked across all stored attention and output fields without changing its tolerances.

`tests/fixtures/delivery_baseline.json` records the pre-edit source/runtime hashes,
checkpoint/fixture identity, environment, and baseline outcomes. Before delivery edits,
40 offline tests passed and two optional live-NATTEN checks skipped; the existing
setuptools wheel-from-sdist installed successfully. Baseline dependencies came from the
existing Python 3.11 environment, including an editable sibling madmom-infer checkout;
that run does not claim independent published-dependency resolution.

## Installed-wheel verification

Build with `python -m build` (wheel from sdist), install the resulting wheel into an
environment with its dependencies, and run from outside the checkout:

```bash
cd /tmp
python /path/to/all-in-one-infer/tools/installed_smoke.py \
  /path/to/all-in-one-infer/tests/fixtures/neighborhood_attention_golden.pt \
  /path/to/all-in-one-infer/tests/fixtures/delivery_baseline.json
all-in-one-infer --help
```

This verifies installed module location, production `.py`/`.toml` bytes, checkpoint
configuration, lazy public session construction/close, and all 24 stored NATTEN attention
and output fields at the original `atol=rtol=1e-5`. It reports absolute and scale-relative
errors. No model checkpoint or external audio is needed. The reusable verification workflow
runs this check on Python 3.9–3.12 before the separate publishing job may run.

The two approved production-file exceptions are the Python 3.9 annotation fixes in
`checkpoints.py` and `session.py`. Each installed module must match final source exactly;
removing precisely its one new `from __future__ import annotations` import must restore its
immutable original SHA-256. Every other production file is compared directly to its original hash. The actual
Python 3.9 preflight failed at `Path | str | None` before this fix; import, custom configuration,
metadata override, and malformed-configuration checks then passed on that interpreter. That
checkpoint-only preflight missed the session's evaluated `Optional[str | Path]` annotation;
the full hosted Python 3.9 run exposed it and now also gates public package/session imports.
