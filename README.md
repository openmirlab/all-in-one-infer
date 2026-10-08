# All-In-One-Infer

[![Visual Demo](https://img.shields.io/badge/Visual-Demo-8A2BE2)](https://taejun.kim/music-dissector/)
[![arXiv](https://img.shields.io/badge/arXiv-2307.16425-B31B1B)](http://arxiv.org/abs/2307.16425/)

Music structure analysis that installs with one command on modern PyTorch. Given an audio
file, it predicts **tempo (BPM), beats, downbeats, and functional segments** (intro, verse,
chorus, bridge, outro, …) using the unchanged models and algorithms of
[All-In-One](https://github.com/mir-aidj/all-in-one) by Taejun Kim and Juhan Nam.

```bash
pip install "all-in-one-infer @ git+https://github.com/openmirlab/all-in-one-infer.git"
```

This inference-only fork repackages the original so it installs cleanly: a pure-PyTorch
neighborhood attention replaces the compiled NATTEN extension, and source separation and
spectrogram/beat decoding come from the sibling packages
[demucs-infer](https://github.com/openmirlab/demucs-infer) and
[madmom-infer](https://github.com/openmirlab/madmom-infer). Results, model names, and the
JSON format match upstream. Formerly published as `all-in-one-fix`; see
[Migrating](#migrating).

## Installation

Install from GitHub; new versions are no longer published to PyPI.

```bash
pip install "all-in-one-infer @ git+https://github.com/openmirlab/all-in-one-infer.git"
# or
uv add "all-in-one-infer @ git+https://github.com/openmirlab/all-in-one-infer.git"
```

- **Requirements:** Python 3.9+, PyTorch 2.0+ (no upper bound), Linux/macOS/Windows.
  demucs-infer and madmom-infer install automatically from pinned Git revisions.
- **GPU:** install a CUDA build of PyTorch first, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- **MP3 and other non-WAV/FLAC input** needs the `ffmpeg`/`ffprobe` executables on your
  `PATH` (`sudo apt install ffmpeg` or `brew install ffmpeg`); see [MP3 input](#mp3-input).
- **Check the install:** `all-in-one-infer --help`.

## Quick start

```shell
all-in-one-infer song1.wav song2.mp3        # writes ./struct/song1.json, ./struct/song2.json
```

```python
import allin1_infer

result = allin1_infer.analyze('song.wav')    # returns results; writes nothing unless out_dir=
print(result.bpm, result.beats[:4], result.segments[0])
```

Both run the same pipeline: source separation (HTDemucs) → spectrograms → the Harmonix
structure model → postprocessing. A result looks like:

```python
AnalysisResult(
  path='/path/to/song.wav',
  bpm=100,
  beats=[0.33, 0.75, 1.14, ...],
  downbeats=[0.33, 1.94, 3.53, ...],
  beat_positions=[1, 2, 3, 4, 1, 2, 3, 4, 1, ...],
  segments=[Segment(start=0.0, end=0.33, label='start'),
            Segment(start=0.33, end=13.13, label='intro'),
            Segment(start=13.13, end=37.53, label='chorus'), ...],
)
```

The CLI's JSON files contain the same fields; load one back with
`allin1_infer.load_result('./struct/song.json')`.

## Python API

### `analyze()`

```python
results = allin1_infer.analyze(['song1.wav', 'song2.mp3'], out_dir='./struct')
```

A single path returns one `AnalysisResult`; a list returns a list. Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `out_dir` | `None` | Save JSON results here; existing results are reused unless `overwrite=True` |
| `model` | `'harmonix-all'` | See [Models](#models) |
| `device` | CUDA if available, else CPU | Also accepts `'auto'`, `'cpu'`, `'cuda'`, `'cuda:N'`; `'mps'` is rejected |
| `visualize`, `sonify` | `False` | `True` saves to `./viz` / `./sonif`, or pass a directory |
| `include_activations`, `include_embeddings` | `False` | Add raw model outputs; see [Research outputs](#research-outputs) |
| `keep_byproducts`, `demix_dir`, `spec_dir` | `False`, `./demix`, `./spec` | Keep separated stems and spectrograms |
| `stem_provider`, `stems_dict`, `stems_input`, `skip_separation` | — | Supply your own stems; see [Using your own stems](#using-your-own-stems) |
| `compile_model` | `False` | Experimental `torch.compile`: ~57 s one-time cost, ~38% faster forward afterwards |
| `demucs_overlap`, `demucs_fp16` | `0.25`, `False` | Experimental; changing them can shift segment boundaries |

`analyze()` loads models on every call. To process many tracks in one process, use a session.

### `AllInOneSession`

A session keeps the Harmonix model and one HTDemucs separator loaded between calls. The
separator loads on the first mixed-audio call; direct stems input never loads it.

```python
from allin1_infer import AllInOneSession

with AllInOneSession(model='harmonix-all', device='cuda') as session:   # load() on enter
    for path in ['song1.wav', 'song2.wav']:
        result = session.infer(path)                                   # same options as analyze()
    print(session.status, session.cache_info())
```

`infer()` requires a loaded session. `release()` frees the models and allows a later
`load()`; `close()` is final. `checkpoint_config` / `checkpoint_overrides` replace the
package's checkpoint URLs (`config/checkpoints.toml`).

### Visualization and sonification

```python
fig = allin1_infer.visualize(result)          # matplotlib Figure (list in, list out)
y, sr = allin1_infer.sonify(result)           # audio with clicks on beats/downbeats and boundary cues
```

Both accept `out_dir=` to save files. From the CLI, `-v` writes PDFs to `./viz` and `-s`
writes `.sonif.wav` files to `./sonif`.

![Visualization](./assets/viz.png)

Try it online at the [Hugging Face Space](https://huggingface.co/spaces/taejunkim/all-in-one).

## Using your own stems

The model analyzes four stems (bass, drums, other, vocals). By default HTDemucs produces them;
you can supply them instead.

```python
from allin1_infer import (analyze, StemsInput, create_stems_input_from_directory,
                          PrecomputedStemProvider, CustomSeparatorProvider)

# Stems already on disk: skip separation entirely (Demucs is never loaded)
analyze(stems_input=StemsInput(bass='b.wav', drums='d.wav', other='o.wav',
                               vocals='v.wav', identifier='my_song'))
analyze(stems_input=[create_stems_input_from_directory('song1_stems/'),
                     create_stems_input_from_directory('song2_stems/')])

# Map each mix to a directory of precomputed stems (any separation tool)
analyze(['song1.wav'], stem_provider=PrecomputedStemProvider({'song1.wav': '/stems/song1/'}))

# Your own separator: an object whose separate(audio_path, output_dir, device) writes
# bass/drums/other/vocals.wav and returns that directory
analyze(['song.wav'], stem_provider=CustomSeparatorProvider(my_separator))
```

The CLI offers the same modes:

```shell
all-in-one-infer --stems-from-dir ./my_stems --stems-id my_song        # {bass,drums,other,vocals}.wav
all-in-one-infer --stems-from-dir ./stems --stems-pattern "track_{stem}.wav"
all-in-one-infer --stems-bass b.wav --stems-drums d.wav --stems-other o.wav --stems-vocals v.wav
all-in-one-infer song1.wav song2.wav --stems-dict stems_mapping.json   # {"song1.wav": "/stems/song1/", ...}
all-in-one-infer song.wav --skip-separation --demix-dir ./existing_stems
```

## CLI

`all-in-one-infer -h` lists every option. The most used:

| Option | Meaning |
|---|---|
| `-o, --out-dir` | Results directory (default `./struct`) |
| `-m, --model`, `-d, --device` | Model name and device |
| `-v`, `-s` | Save visualizations / sonifications |
| `-a, --activ`, `-e, --embed` | Save raw activations / embeddings |
| `-k, --keep-byproducts` | Keep separated stems (`./demix`) and spectrograms (`./spec`) |
| `--overwrite` | Re-analyze tracks that already have results |
| `--cache-info`, `--clear-cache`, `--clear-cache-dry-run` | Inspect or free the downloaded checkpoints |

## Models

The models are trained on the [Harmonix Set](https://github.com/urinieto/harmonixset) with
8-fold cross-validation ([paper](http://arxiv.org/abs/2307.16425)).

- `harmonix-all` (default): averages the 8 fold models.
- `harmonix-fold0` … `harmonix-fold7`: a single fold model, faster.

Checkpoints download on first use from the authors' Hugging Face repository
([taejunkim/allinone](https://huggingface.co/taejunkim/allinone)), are SHA-256 verified, and
are cached in torch hub's `checkpoints` directory (by default
`~/.cache/torch/hub/checkpoints/`) next to the Demucs weights. Inspect or clear that cache with
`--cache-info` / `--clear-cache`, or `allin1_infer.print_cache_info()`,
`get_cache_size()`, `list_cached_models()` and `clear_model_cache(dry_run=True)`.

**Speed:** on an RTX 4090 with an i9-10940X, `harmonix-all` processed 10 songs (33 minutes of
audio) in 73 seconds; the structure model's own forward pass takes under 0.5 s per song on
that GPU, so most of the time goes to source separation.

## Research outputs

Frame-level outputs before postprocessing, at `result.activation_fps` frames per second
(100 for the released models):

- **Activations** (`--activ` → `<name>.activ.npz`, or `include_activations=True`): `beat`,
  `downbeat`, and `segment` sigmoid outputs of shape `[time]`, and the `label` softmax of
  shape `[10, time]` over `allin1_infer.HARMONIX_LABELS`
  (`start, end, intro, outro, break, bridge, inst, solo, verse, chorus`).
- **Embeddings** (`--embed` → `<name>.embed.npy`, or `include_embeddings=True`): shape
  `[stems=4, time, 24]` with stems ordered bass, drums, other, vocals; `harmonix-all` stacks
  the 8 models as a trailing dimension.

## MP3 input

WAV and FLAC are read with `soundfile`. Every other format is decoded by FFmpeg, and loading
fails with a clear error when `ffmpeg`/`ffprobe` are not on `PATH`; no other decoder is
substituted. Different MP3 decoders can shift audio by 20–40 ms, which matters for beat
tracking (the usual tolerance is 70 ms), so for datasets convert to WAV once and analyze the
WAV files:

```shell
ffmpeg -i song.mp3 song.wav
```

## Migrating

| | Import | CLI |
|---|---|---|
| Upstream All-In-One | `from allin1 import analyze` | `allin1` |
| all-in-one-fix (before 3.0.0) | `from allin1fix import analyze` | `allin1fix` |
| **all-in-one-infer** | `from allin1_infer import analyze` | `all-in-one-infer` |

Function signatures, model names, result fields, and the JSON format are unchanged. The
package needs neither NATTEN nor the old `demucs` package.

## Scope

Inference on the pretrained `harmonix-*` models, stem handling around it, visualization,
sonification, and research outputs. Permanently out of scope:

- **Training.** Use upstream [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one).
- **Reimplementing the sibling packages.** Separation stays in demucs-infer and
  spectrogram/beat decoding in madmom-infer.
- **NATTEN.** Its releases that provide the API these checkpoints need build only against
  torch < 2.8. The pure-PyTorch replacement reproduces NATTEN 0.17.5's output on recorded
  fixtures and is the only backend.
- **Bundled or altered weights.** No checkpoint is committed to git or shipped in a
  package; all are downloaded from their original hosts, unmodified.

## Development

```bash
git clone https://github.com/openmirlab/all-in-one-infer.git && cd all-in-one-infer
pip install -e ".[dev]"
pytest tests/ -v                     # offline contracts and golden fixtures, no downloads
pytest tests/ -v -m integration --run-integration --integration-audio /path/to/music.wav
```

CI runs the offline suite on Python 3.9–3.12 with CPU PyTorch and without TorchAudio, then
builds a wheel and checks the installed package. [CLAUDE.md](CLAUDE.md) holds the
conventions and verification commands; [tools/README.md](tools/README.md) covers
installed-wheel and upstream-comparison checks; [CHANGELOG.md](CHANGELOG.md) holds the
history.

## Credits and citation

All credit for the models, algorithms, and pretrained weights belongs to
[All-In-One](https://github.com/mir-aidj/all-in-one) by [Taejun Kim](https://taejun.kim/) and
Juhan Nam (KAIST). Source separation uses [Demucs](https://github.com/facebookresearch/demucs)
by Alexandre Défossez / Meta AI through demucs-infer. This repository contributes the
packaging and tooling.

```bibtex
@inproceedings{taejun2023allinone,
  title={All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={Défossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## License

MIT, like All-In-One and Demucs. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Report issues at
[GitHub Issues](https://github.com/openmirlab/all-in-one-infer/issues).
