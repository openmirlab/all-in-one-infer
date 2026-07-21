# All-In-One-Infer Music Structure Analyzer

[![Visual Demo](https://img.shields.io/badge/Visual-Demo-8A2BE2)](https://taejun.kim/music-dissector/)
[![arXiv](https://img.shields.io/badge/arXiv-2307.16425-B31B1B)](http://arxiv.org/abs/2307.16425/)
[![PyPI](https://img.shields.io/pypi/v/all-in-one-infer)](https://pypi.org/project/all-in-one-infer/)

**An enhanced version of All-In-One with integrated source separation and modern PyTorch compatibility**

Available on PyPI: [https://pypi.org/project/all-in-one-infer/](https://pypi.org/project/all-in-one-infer/)

> ## Officially renamed: all-in-one-fix → all-in-one-infer
>
> This project has **officially moved** from `all-in-one-fix` to **`all-in-one-infer`** (as of v3.0.0, 2026-07).
>
> - **Install:** `pip install all-in-one-infer` (the old `all-in-one-fix` PyPI package will no longer receive updates)
> - **Import:** `import allin1_infer` (formerly `allin1fix`)
> - **CLI:** `all-in-one-infer` (formerly `allin1fix`)
>
> Existing `all-in-one-fix` users: see the [Migration](#migration-from-all-in-one) section below.

This package predicts, given an audio file:
1. Tempo (BPM)
2. Beats
3. Downbeats
4. Functional segment boundaries
5. Functional segment labels (e.g., intro, verse, chorus, bridge, outro)

**Table of Contents**

- [Why This Exists](#why-this-exists)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Features](#features)
- [Scope](#scope)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage for CLI](#usage-for-cli)
- [Usage for Python](#usage-for-python)
- [Visualization & Sonification](#visualization--sonification)
- [Available Models](#available-models)
- [Speed](#speed)
- [Advanced Usage for Research](#advanced-usage-for-research)
- [Concerning MP3 Files](#concerning-mp3-files)
- [Migration from All-In-One](#migration-from-all-in-one)
- [What This Project Will Never Bundle](#what-this-project-will-never-bundle)
- [Development](#development)
- [License](#license)
- [Support](#support)
- [Documentation](#documentation)

## Why This Exists

The original **[All-In-One](https://github.com/mir-aidj/all-in-one)** is an
excellent music structure analysis tool — predicting tempo, beats, downbeats,
and functional segments in a single pass — but it became hard to install in
modern environments:

1. **PyTorch 2.x compatibility**: the original depends on NATTEN, a
   source-only CUDA/C++ extension that must compile against the exact
   installed torch version at install time. NATTEN's newer releases also
   dropped the legacy API and relative-positional-bias support the
   pretrained checkpoints require, and NATTEN never supported some
   platforms (e.g. macOS / Apple Silicon) at all.
2. **Source separation dependency**: the original required a separate,
   by-then-unmaintained `demucs` install (PyTorch 1.x only), which pulled
   in its own version conflicts.
3. **Performance**: no model caching — every run reloaded the ensemble's 8
   checkpoints from scratch.
4. **Packaging**: `madmom`, used internally for spectrogram/beat-decoding,
   has never published a PyPI release with a working sdist and required a
   `pip install git+https://...` step (git installs are also rejected by
   PyPI metadata rules).

**What this repo reprovides:** the exact same research — model
architectures, beat/downbeat/tempo algorithms, and structure segmentation
are all unchanged from upstream — packaged so `pip install all-in-one-infer`
just works: pure-PyTorch neighborhood attention (no compiler, no NATTEN
requirement), source separation via the sibling
[demucs-infer](https://github.com/openmirlab/demucs-infer) package, and
spectrogram/beat-decoding via the sibling
[madmom-infer](https://github.com/openmirlab/madmom-infer) package — both
plain PyPI installs with no git-install step. See
[CHANGELOG.md](docs/CHANGELOG.md) for the full version-by-version history of
how this fork got here.

## Acknowledgments

All-In-One-Infer builds entirely on the research, model architectures, and
pretrained weights of the upstream project below. This repo's own
contribution is limited to packaging, PyTorch 2.x compatibility, and
performance/workflow tooling around that unchanged research — all credit
for the core algorithms belongs to the original authors.

- **Upstream project:** [All-In-One](https://github.com/mir-aidj/all-in-one) (`mir-aidj` on GitHub)
- **Individual authors:** [Taejun Kim](https://taejun.kim/) and Juhan Nam (KAIST)
- **Source repository:** [github.com/mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)
- **Pretrained weights host:** [huggingface.co/taejunkim/allinone](https://huggingface.co/taejunkim/allinone) — this package downloads the `harmonix-*` checkpoints directly from that Hugging Face repository (via `huggingface_hub.hf_hub_download`) and hosts no mirror

This package also depends on two sibling packages for parts of its
pipeline, each with its own upstream lineage and Acknowledgments section:

- **[demucs-infer](https://github.com/openmirlab/demucs-infer)** — inference-only source separation, itself built on [Demucs](https://github.com/facebookresearch/demucs) by Alexandre Défossez and Meta AI Research
- **[madmom-infer](https://github.com/openmirlab/madmom-infer)** — a from-scratch, pure numpy/scipy reimplementation of the `madmom` spectrogram/DBN-decoding surface this project uses, published to PyPI

## Citation

If you use this package for your research, please cite the following papers.

**All-In-One (core music structure analysis algorithms):**
```bibtex
@inproceedings{taejun2023allinone,
  title={All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
```

**Demucs (source separation models):**
```bibtex
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={Défossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## Features

- **Modern PyTorch support**: compatible with PyTorch 2.0 through 2.7+ and CUDA 11.7-12.8, no compiled extensions required by default
- **Pure-PyTorch neighborhood attention**: numerically identical to NATTEN 0.17.5 (golden-fixture tested); optional `[natten]` extra for NATTEN's fused-kernel backend as a speed optimization
- **Integrated source separation**: uses [demucs-infer](https://github.com/openmirlab/demucs-infer) with intelligent model caching (~6x faster on repeated use) and automatic GPU memory cleanup
- **Flexible stems input**: custom separation models via a pluggable provider interface, pre-computed stems from any tool, direct stems input (skip separation entirely), and hybrid workflows mixing all three
- **Cache management**: `--cache-info` / `--clear-cache` CLI flags and a matching Python API to inspect and free the multi-GB separation-model cache
- **Reusable lifecycle**: `AllInOneSession` provides explicit `load()`, ready-only `infer()`, `release()`, `close()`, `status`, `cache_info()`, and context-manager support; mixed-input calls reuse one resident Harmonix + HTDemucs pipeline, while legacy `analyze()` remains lazy and compatible
- **PyPI-only install**: as of 3.1.0, `madmom` is replaced by [madmom-infer](https://github.com/openmirlab/madmom-infer) — nothing to compile, nothing to `git+https://` install
- **100% backward compatible**: same analysis JSON structure, function signatures, model names, and accuracy as upstream All-In-One

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the detailed, version-by-version history of how these features were added.

## Scope

**In scope:**
- Inference on the pretrained `harmonix-*` models: tempo, beats, downbeats, functional segment boundaries and labels
- Source separation orchestration (via demucs-infer) as a preprocessing step, including flexible/custom/pre-computed stems input
- Visualization and sonification of analysis results
- Frame-level raw activations and embeddings for research use
- Packaging, PyTorch 2.x compatibility, and performance tooling (caching, GPU cleanup) around the unchanged upstream research

**Out of scope, forever:**
- **Training.** This package is inference-only; training code has been removed and will not return. To retrain models, use the upstream [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) repository directly (see [Training](#training) below).
- **Reimplementing demucs-infer or madmom-infer's internals here.** Source separation and spectrogram/beat-decoding are deliberately delegated to those sibling packages rather than vendored, so accuracy and maintenance responsibility stay with the package that owns them.
- **A NATTEN-required install path.** Pure-PyTorch neighborhood attention is the default and will remain so; NATTEN stays an optional, non-default speed extra.

## Installation

Available on PyPI: [https://pypi.org/project/all-in-one-infer/](https://pypi.org/project/all-in-one-infer/)

### Quick Install from PyPI

```bash
pip install all-in-one-infer
```

**Or if you prefer UV (faster):**
```bash
uv add all-in-one-infer
```

That's it — no `--no-build-isolation`, no "install torch first", no separate
git-install step. Since NATTEN is no longer a dependency and, as of 3.1.0,
madmom is replaced by [madmom-infer](https://github.com/openmirlab/madmom-infer)
(a plain PyPI package), there is nothing to compile and nothing to fetch from
git at install time.

### Requirements

- **Python**: 3.9 or later (required for `scipy>=1.13` and `madmom-infer`)
- **PyTorch**: 2.0.0 or later (no upper bound)
- **OS**: Linux, macOS, Windows

### GPU Support (Optional)

For GPU acceleration, install PyTorch with CUDA support:

```bash
# Example: CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install all-in-one-infer
```

### Optional: NATTEN fused kernels

Neighborhood attention runs on a built-in pure-PyTorch implementation that is
numerically identical to NATTEN. If you want NATTEN's fused CUDA kernels as a
speed optimization (Linux + CUDA + torch < 2.8 only, compiles at install time):

```bash
pip install torch"<2.8"
pip install "all-in-one-infer[natten]" --no-build-isolation
```

It is picked up automatically when importable; otherwise the pure-PyTorch
backend is used. Results are identical either way.

### Verify Installation

After installation, verify it worked:

```bash
# Check if installed
python -c "import allin1_infer; print('allin1_infer installed successfully!')"

# Check version
python -c "import allin1_infer; print(allin1_infer.__version__)"

# Test CLI
all-in-one-infer --help
```

### Troubleshooting

**Installation fails with scipy version error**
- Cause: Using Python < 3.9
- Solution: Ensure Python 3.9+ is used

**ImportError: No module named 'madmom_infer'**
- Cause: `madmom-infer` failed to install as a regular dependency (rare — it's a plain PyPI package with no compiled extensions)
- Solution: Run `pip install madmom-infer` before using all-in-one-infer

### Installation from GitHub (Development)

If you want to install the latest development version from GitHub:

**Using UV (Recommended):**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install from GitHub
uv pip install git+https://github.com/openmirlab/all-in-one-infer.git
```

**Using pip:**
```bash
pip install git+https://github.com/openmirlab/all-in-one-infer.git
```

**Development Installation (Editable):**
```bash
git clone https://github.com/openmirlab/all-in-one-infer.git
cd all-in-one-infer
pip install -e .
```

**Note:** All dependencies (including **demucs-infer** and **madmom-infer**) are installed automatically from PyPI — no separate git-install step.

### (Optional) Install FFmpeg for MP3 support

For Ubuntu:

```shell
sudo apt install ffmpeg
```

For macOS:

```shell
brew install ffmpeg
```

## Quick Start

```shell
# Analyze one or more audio files from the CLI
all-in-one-infer your_audio_file.wav
```

```python
# Or from Python
from allin1_infer import analyze

result = allin1_infer.analyze('your_audio_file.wav')
print(result.bpm, result.beats[:4], result.segments[0])
```

Both paths run the same pipeline: source separation (via demucs-infer) →
spectrogram + beat/downbeat extraction (via madmom-infer) → model inference
→ postprocessing. Results are saved as JSON under `./struct` by default when
using the CLI; the Python `analyze()` function returns results in memory and
only writes to disk if you pass `out_dir=`. See [Usage for CLI](#usage-for-cli)
and [Usage for Python](#usage-for-python) below for the full option set.

## Usage for CLI

### Basic Usage

To analyze audio files:
```shell
all-in-one-infer your_audio_file1.wav your_audio_file2.mp3
```

### New Stems Features

**1. Direct stems input from directory:**
```shell
all-in-one-infer --stems-from-dir ./my_stems --stems-id "my_song" -o ./results
# Expects: ./my_stems/{bass,drums,other,vocals}.wav
```

**2. Custom stem filename patterns:**
```shell
all-in-one-infer --stems-from-dir ./stems --stems-pattern "track_{stem}.wav" -o ./results
# Expects: ./stems/track_{bass,drums,other,vocals}.wav
```

**3. Individual stem files:**
```shell
all-in-one-infer \
  --stems-bass path/to/bass.wav \
  --stems-drums path/to/drums.wav \
  --stems-other path/to/other.wav \
  --stems-vocals path/to/vocals.wav \
  --stems-id "my_track" -o ./results
```

**4. Pre-computed stems mapping:**
```shell
# Create stems_mapping.json:
{
  "song1.wav": "/path/to/song1_stems/",
  "song2.wav": "/path/to/song2_stems/"
}

all-in-one-infer song1.wav song2.wav --stems-dict stems_mapping.json -o ./results
```

**5. Skip separation (use existing stems in demix-dir):**
```shell
all-in-one-infer track.wav --skip-separation --demix-dir ./existing_stems -o ./results
```
Results will be saved in the `./struct` directory by default:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file2.json
```
The analysis results will be saved in JSON format:
```json
{
  "path": "/path/to/your_audio_file.wav",
  "bpm": 100,
  "beats": [ 0.33, 0.75, 1.14, ... ],
  "downbeats": [ 0.33, 1.94, 3.53, ... ],
  "beat_positions": [ 1, 2, 3, 4, 1, 2, 3, 4, 1, ... ],
  "segments": [
    {
      "start": 0.0,
      "end": 0.33,
      "label": "start"
    },
    {
      "start": 0.33,
      "end": 13.13,
      "label": "intro"
    },
    {
      "start": 13.13,
      "end": 37.53,
      "label": "chorus"
    },
    {
      "start": 37.53,
      "end": 51.53,
      "label": "verse"
    },
    ...
  ]
}
```

### Cache Management

Separation models are downloaded to `~/.cache/torch/hub/checkpoints/` and can use several GB of disk space.

**View cache information:**
```shell
all-in-one-infer --cache-info
```

Output:
```
============================================================
Model Cache Information
============================================================
Cache directory: /home/user/.cache/torch/hub/checkpoints
Total size: 3.19 GB
Number of models: 23

Cached models:
------------------------------------------------------------
  7fd6ef75-a905dd85.th                        37.61 MB  2025-09-08 12:20:50
  14fc6a69-a89dd0ee.th                        36.71 MB  2025-09-08 12:20:46
  ...
============================================================
```

**Preview what would be deleted (dry run):**
```shell
all-in-one-infer --clear-cache-dry-run
```

**Clear all cached models:**
```shell
all-in-one-infer --clear-cache
```

**Python API:**
```python
import allin1_infer

# View cache info
allin1_infer.print_cache_info()

# Get cache size
size_gb = allin1_infer.get_cache_size()

# List models
models = allin1_infer.list_cached_models()

# Clear cache (dry run first!)
count = allin1_infer.clear_model_cache(dry_run=True)
count = allin1_infer.clear_model_cache()  # Actually delete
```

### Technical Improvements

All-In-One-Infer includes several technical enhancements over the original:

- **Modern PyTorch Support**: Compatible with PyTorch 2.x and CUDA 12.x
- **Pure-PyTorch Neighborhood Attention**: NATTEN is no longer required — a numerically identical pure-PyTorch implementation ships by default; NATTEN 0.17.x can optionally be installed via the `[natten]` extra as a faster fused-kernel backend
- **Source Separation**: Uses demucs-infer package with model caching and GPU cleanup
- **Memory Optimization**: Automatic GPU memory cleanup prevents OOM errors on batch processing
- **Performance**: 6x faster on repeated use with intelligent model caching
- **Error Handling**: Better error messages with fuzzy matching and helpful suggestions
- **Modular Architecture**: Clean separation of concerns for easier maintenance and extension
- **Cache Management**: Built-in tools to view and manage cached separation models

### All Available CLI Options

```shell
$ all-in-one-infer -h

usage: all-in-one-infer [-h] [-o OUT_DIR] [-v] [--viz-dir VIZ_DIR] [-s]
                 [--sonif-dir SONIF_DIR] [-a] [-e] [-m MODEL] [-d DEVICE] [-k]
                 [--demix-dir DEMIX_DIR] [--spec-dir SPEC_DIR] [--overwrite]
                 [--no-multiprocess] [--compile-model] [--demucs-overlap DEMUCS_OVERLAP]
                 [--demucs-fp16] [--stems-dict STEMS_DICT]
                 [--stems-dir STEMS_DIR] [--skip-separation] [--no-demucs]
                 [--stems-bass STEMS_BASS] [--stems-drums STEMS_DRUMS]
                 [--stems-other STEMS_OTHER] [--stems-vocals STEMS_VOCALS]
                 [--stems-from-dir STEMS_FROM_DIR]
                 [--stems-pattern STEMS_PATTERN] [--stems-id STEMS_ID]
                 [paths ...]

positional arguments:
  paths                 Path to tracks (for single track mode) or omit for
                        stems input mode

Core Options:
  -o, --out-dir         Path to store analysis results (default: ./struct)
  -v, --visualize       Save visualizations (default: False)
  -s, --sonify          Save sonifications (default: False)
  -m, --model          Model to use (default: harmonix-all)
  -d, --device         Device to use (default: cuda if available else cpu)
  -k, --keep-byproducts Keep demixed audio and spectrograms (default: False)

Performance Options (opt-in, experimental):
  --compile-model       torch.compile the model(s) (reduce-overhead mode).
                        ~57s one-time compile cost, ~38% faster steady-state
                        forward -- only worth it for long-lived/batch
                        processing (default: False)
  --demucs-overlap DEMUCS_OVERLAP
                        Accuracy-affecting: overlap fraction for demucs
                        separation chunks. Default 0.25 matches demucs' own
                        default (unchanged behavior); other values can shift
                        segment boundaries
  --demucs-fp16         Accuracy-affecting: run demucs separation under
                        fp16 autocast (CUDA only) for faster separation
                        (default: False)

Stems Input Options:
  --stems-dict         JSON file mapping audio paths to stem directories
  --stems-from-dir     Directory containing bass.wav, drums.wav, other.wav, vocals.wav
  --stems-pattern      Pattern for stem files (e.g. "song_{stem}.wav")
  --stems-bass         Path to bass stem file
  --stems-drums        Path to drums stem file  
  --stems-other        Path to other stem file
  --stems-vocals       Path to vocals stem file
  --stems-id           Identifier for the stem set
  --skip-separation    Skip source separation, use existing stems
```

## Usage for Python

### Reusable model session

Use an explicit session when several mixed-input tracks share one loaded
pipeline. `load()` prepares the Harmonix structure model and creates one owned
HTDemucs separator; the separator loads lazily on the first mixed-input call,
then remains resident for reuse. `release()` frees both components without
deleting cached files. Direct stems input bypasses Demucs entirely, including
model loading. `config/checkpoints.toml` is package-owned and may be replaced
through `checkpoint_config` or generic checkpoint overrides.

```python
from allin1_infer import AllInOneSession

with AllInOneSession(model="harmonix-all", device="cuda") as session:
    result = session.infer("song.wav")
    info = session.cache_info()
    print(session.status, info["cache_dir"], info["components"])
```

`session.infer()` requires a ready session. For one-shot use, keep calling
`allin1_infer.analyze(...)`; it continues to load lazily with the historical
signature and output format.
`release()` permits a later reload, while `close()` is terminal. Device
requests preserve legacy `None`/`auto` selection and accept explicit `cpu`,
`cuda`, `cuda:N`, or supported `mps`; malformed or unavailable explicit
requests raise before loading either Harmonix or Demucs.

### Basic Usage

```python
from allin1_infer import analyze

# Analyze audio files (uses demucs-infer for separation)
results = analyze(['song1.wav', 'song2.mp3'])
```

### New Stems API Features

**1. Custom separation models:**
```python
from allin1_infer import analyze, CustomSeparatorProvider

class MyCustomSeparator:
    def __init__(self, model_path):
        self.model = load_my_model(model_path)
    
    def separate(self, audio_path, output_dir, device):
        # Your separation logic here
        # Must return Path to directory with bass.wav, drums.wav, other.wav, vocals.wav
        stems_dir = output_dir / 'my_model' / audio_path.stem
        stems_dir.mkdir(parents=True, exist_ok=True)
        
        # Your model inference
        stems = self.model.separate(audio_path, device)
        
        # Save stems
        for stem_name, audio_data in stems.items():
            save_audio(audio_data, stems_dir / f"{stem_name}.wav")
        
        return stems_dir

# Use your custom model
separator = MyCustomSeparator("path/to/model.pth")
provider = CustomSeparatorProvider(separator)
results = analyze(['song.wav'], stem_provider=provider)
```

**2. Pre-computed stems:**
```python
from allin1_infer import analyze, PrecomputedStemProvider

# Use stems from any source separation tool
stems_mapping = {
    'song1.wav': '/path/to/spleeter_output/song1/',
    'song2.wav': '/path/to/mdx_output/song2/',
    'song3.wav': '/path/to/custom_stems/song3/'
}
provider = PrecomputedStemProvider(stems_mapping)
results = analyze(['song1.wav', 'song2.wav', 'song3.wav'], stem_provider=provider)
```

**3. Direct stems input:**
```python
from allin1_infer import analyze, StemsInput, create_stems_input_from_directory

# Method 1: Manual specification
stems = StemsInput(
    bass='path/to/bass.wav',
    drums='path/to/drums.wav', 
    other='path/to/other.wav',
    vocals='path/to/vocals.wav',
    identifier='my_song'
)

# Method 2: From directory (expects bass.wav, drums.wav, other.wav, vocals.wav)
stems = create_stems_input_from_directory('/path/to/stems_folder')

# Method 3: Multiple tracks with different stems
stems_list = [
    create_stems_input_from_directory('/path/to/song1_stems'),
    create_stems_input_from_directory('/path/to/song2_stems')
]

results = analyze(stems_input=stems_list)
```

**4. Hybrid workflows:**
```python
# Mix different approaches in the same analysis
from allin1_infer import analyze, PrecomputedStemProvider, StemsInput

# Some tracks have pre-computed stems
stems_mapping = {'song1.wav': '/path/to/stems/'}
provider = PrecomputedStemProvider(stems_mapping)

# Other tracks use default separation  
regular_tracks = ['song2.wav', 'song3.wav']

# Process each group
results1 = analyze(['song1.wav'], stem_provider=provider)
results2 = analyze(regular_tracks)  # Uses default HTDemucs
```

Available functions:
- [`analyze()`](#analyze)
- [`load_result()`](#load_result)
- [`visualize()`](#visualize)
- [`sonify()`](#sonify)

### `analyze()`
Analyzes the provided audio files and returns the analysis results.

```python
import allin1_infer

# You can analyze a single file:
result = allin1_infer.analyze('your_audio_file.wav')

# Or multiple files:
results = allin1_infer.analyze(['your_audio_file1.wav', 'your_audio_file2.mp3'])
```
A result is a dataclass instance containing:
```python
AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100,
  beats=[0.33, 0.75, 1.14, ...],
  beat_positions=[1, 2, 3, 4, 1, 2, 3, 4, 1, ...],
  downbeats=[0.33, 1.94, 3.53, ...], 
  segments=[
    Segment(start=0.0, end=0.33, label='start'), 
    Segment(start=0.33, end=13.13, label='intro'), 
    Segment(start=13.13, end=37.53, label='chorus'), 
    Segment(start=37.53, end=51.53, label='verse'), 
    Segment(start=51.53, end=64.34, label='verse'), 
    Segment(start=64.34, end=89.93, label='chorus'), 
    Segment(start=89.93, end=105.93, label='bridge'), 
    Segment(start=105.93, end=134.74, label='chorus'), 
    Segment(start=134.74, end=153.95, label='chorus'), 
    Segment(start=153.95, end=154.67, label='end'),
  ]),
```
Unlike CLI, it does not save the results to disk by default. You can save them as follows:
```python
result = allin1_infer.analyze(
  'your_audio_file.wav',
  out_dir='./struct',
)
```

#### Parameters:

- `paths` : `Union[PathLike, List[PathLike]]`  
List of paths or a single path to the audio files to be analyzed.
  
- `out_dir` : `PathLike` (optional)  
Path to the directory where the analysis results will be saved. By default, the results will not be saved.
  
- `visualize` : `Union[bool, PathLike]` (optional)  
Whether to visualize the analysis results or not. If a path is provided, the visualizations will be saved in that directory. Default is False. If True, the visualizations will be saved in './viz'.
  
- `sonify` : `Union[bool, PathLike]` (optional)  
Whether to sonify the analysis results or not. If a path is provided, the sonifications will be saved in that directory. Default is False. If True, the sonifications will be saved in './sonif'.
  
- `model` : `str` (optional)  
Name of the pre-trained model to be used for the analysis. Default is 'harmonix-all'. Please refer to the documentation for the available models.
  
- `device` : `str` (optional)  
Device to be used for computation. Default is 'cuda' if available, otherwise 'cpu'.
  
- `include_activations` : `bool` (optional)  
Whether to include activations in the analysis results or not.
  
- `include_embeddings` : `bool` (optional)  
Whether to include embeddings in the analysis results or not.
  
- `demix_dir` : `PathLike` (optional)  
Path to the directory where the source-separated audio will be saved. Default is './demix'.
  
- `spec_dir` : `PathLike` (optional)  
Path to the directory where the spectrograms will be saved. Default is './spec'.
  
- `keep_byproducts` : `bool` (optional)  
Whether to keep the source-separated audio and spectrograms or not. Default is False.
  
- `multiprocess` : `bool` (optional)  
Whether to use multiprocessing for extracting spectrograms. Default is True.

- `compile_model` : `bool` (optional)  
EXPERIMENTAL. Wrap the loaded model(s) with `torch.compile(mode='reduce-overhead')`. Adds a ~57s one-time compilation cost on first inference, then ~38% faster steady-state forward passes -- only worth it for long-lived/batch processing, not single tracks. Default is False.

- `demucs_overlap` : `float` (optional)  
EXPERIMENTAL, accuracy-affecting. Overlap fraction between chunks passed to demucs' `apply_model()`. Default is 0.25 (demucs' own default; unchanged from prior behavior). Profiling showed different values can shift segment boundaries slightly.

- `demucs_fp16` : `bool` (optional)  
EXPERIMENTAL, accuracy-affecting. Run demucs separation under `torch.autocast('cuda', dtype=torch.float16)` for faster separation. Default is False (fp32, unchanged from prior behavior). Only takes effect on CUDA.

#### Returns:

- `Union[AnalysisResult, List[AnalysisResult]]`  
Analysis results for the provided audio files.


### `load_result()`

Loads the analysis results from the disk.

```python
result = allin1_infer.load_result('./struct/24k_Magic.json')
```


### `visualize()`

Visualizes the analysis results.

```python
fig = allin1_infer.visualize(result)
fig.show()
```

#### Parameters:

- `result` : `Union[AnalysisResult, List[AnalysisResult]]`  
List of analysis results or a single analysis result to be visualized.

- `out_dir` : `PathLike` (optional)  
Path to the directory where the visualizations will be saved. By default, the visualizations will not be saved.

#### Returns:

- `Union[Figure, List[Figure]]`
List of figures or a single figure containing the visualizations. `Figure` is a class from `matplotlib.pyplot`.


### `sonify()`

Sonifies the analysis results.
It will mix metronome clicks for beats and downbeats, and event sounds for segment boundaries
to the original audio file.

```python
y, sr = allin1_infer.sonify(result)
# y: sonified audio with shape (channels=2, samples)
# sr: sampling rate (=44100)
```

#### Parameters:

- `result` : `Union[AnalysisResult, List[AnalysisResult]]`  
List of analysis results or a single analysis result to be sonified.
- `out_dir` : `PathLike` (optional)  
Path to the directory where the sonifications will be saved. By default, the sonifications will not be saved.

#### Returns:

- `Union[Tuple[NDArray, float], List[Tuple[NDArray, float]]]`  
List of tuples or a single tuple containing the sonified audio and the sampling rate.


## Visualization & Sonification
This package provides a simple visualization (`-v` or `--visualize`) and sonification (`-s` or `--sonify`) function for the analysis results.
```shell
all-in-one-infer -v -s your_audio_file.wav
```
The visualizations will be saved in the `./viz` directory by default:
```shell
./viz
└── your_audio_file.pdf
```
The sonifications will be saved in the `./sonif` directory by default:
```shell
./sonif
└── your_audio_file.sonif.wav
```
For example, a visualization looks like this:
![Visualization](./assets/viz.png)

You can try it at [Hugging Face Space](https://huggingface.co/spaces/taejunkim/all-in-one).


## Available Models
The models are trained on the [Harmonix Set](https://github.com/urinieto/harmonixset) with 8-fold cross-validation.
For more details, please refer to the [paper](http://arxiv.org/abs/2307.16425).
* `harmonix-all`: (Default) An ensemble model averaging the predictions of 8 models trained on each fold.
* `harmonix-foldN`: A model trained on fold N (0~7). For example, `harmonix-fold0` is trained on fold 0.

By default, the `harmonix-all` model is used. To use a different model, use the `--model` option:
```shell
all-in-one-infer --model harmonix-fold0 your_audio_file.wav
```


## Speed
With an RTX 4090 GPU and Intel i9-10940X CPU (14 cores, 28 threads, 3.30 GHz),
the `harmonix-all` model processed 10 songs (33 minutes) in 73 seconds.


## Advanced Usage for Research
This package provides researchers with advanced options to extract **frame-level raw activations and embeddings** 
without post-processing. These have a resolution of 100 FPS, equivalent to 0.01 seconds per frame.

### CLI

#### Activations
The `--activ` option also saves frame-level raw activations from sigmoid and softmax:
```shell
$ all-in-one-infer --activ your_audio_file.wav
```
You can find the activations in the `.npz` file:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file1.activ.npz
```
To load the activations in Python:
```python
>>> import numpy as np
>>> activ = np.load('./struct/your_audio_file1.activ.npz')
>>> activ.files
['beat', 'downbeat', 'segment', 'label']
>>> beat_activations = activ['beat']
>>> downbeat_activations = activ['downbeat']
>>> segment_boundary_activations = activ['segment']
>>> segment_label_activations = activ['label']
```
Details of the activations are as follows:
* `beat`: Raw activations from the **sigmoid** layer for **beat tracking** (shape: `[time_steps]`)
* `downbeat`: Raw activations from the **sigmoid** layer for **downbeat tracking** (shape: `[time_steps]`)
* `segment`: Raw activations from the **sigmoid** layer for **segment boundary detection** (shape: `[time_steps]`)
* `label`: Raw activations from the **softmax** layer for **segment labeling** (shape: `[label_class=10, time_steps]`)

The frame rate of these activations (frames per second, i.e. `1 / time_steps` in seconds) is available
as `result.activation_fps` whenever `include_activations=True` -- read it from there instead of hardcoding
100, since it reflects the actual loaded model's `cfg.fps` rather than assuming a fixed value.

You can access the label names as follows:
```python
>>> allin1_infer.HARMONIX_LABELS
['start',
 'end',
 'intro',
 'outro',
 'break',
 'bridge',
 'inst',
 'solo',
 'verse',
 'chorus']
```


#### Embeddings
This package also provides an option to extract raw embeddings from the model.
```shell
$ all-in-one-infer --embed your_audio_file.wav
```
You can find the embeddings in the `.npy` file:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file1.embed.npy
```
To load the embeddings in Python:
```python
>>> import numpy as np
>>> embed = np.load('your_audio_file1.embed.npy')
```
Each model embeds for every source-separated stem per time step, 
resulting in embeddings shaped as `[stems=4, time_steps, embedding_size=24]`:
1. The number of source-separated stems (the order is bass, drums, other, vocals).
2. The number of time steps (frames). The time step is 0.01 seconds (100 FPS).
3. The embedding size of 24.

Using the `--embed` option with the `harmonix-all` ensemble model will stack the embeddings, 
saving them with the shape `[stems=4, time_steps, embedding_size=24, models=8]`.

### Python
The Python API `allin1_infer.analyze()` offers the same options as the CLI:
```python
>>> allin1_infer.analyze(
      paths='your_audio_file.wav',
      include_activations=True,
      include_embeddings=True,
    )

AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100, 
  beats=[...],
  downbeats=[...],
  segments=[...],
  activations={
    'beat': array(...), 
    'downbeat': array(...), 
    'segment': array(...), 
    'label': array(...)
  }, 
  embeddings=array(...),
)
```

## Concerning MP3 Files
Due to variations in decoders, MP3 files can have slight offset differences.
I recommend you to first convert your audio files to WAV format using FFmpeg (as shown below), 
and use the WAV files for all your data processing pipelines.
```shell
ffmpeg -i your_audio_file.mp3 your_audio_file.wav
```
In this package, audio files are read using [demucs-infer](https://github.com/openmirlab/demucs-infer).
To my understanding, Demucs converts MP3 files to WAV using FFmpeg before reading them.
However, using a different MP3 decoder can yield different offsets. 
I've observed variations of about 20~40ms, which is problematic for tasks requiring precise timing like beat tracking, 
where the conventional tolerance is just 70ms. 
Hence, I advise standardizing inputs to the WAV format for all data processing, 
ensuring straightforward decoding.

> **Note (3.1.0+)**: WAV and FLAC inputs are decoded via `soundfile` (bit-
> identical to `torchaudio`, no extra install needed). MP3 (and other lossy
> formats) still decode via `torchaudio`, which requires the separate
> `torchcodec` package on `torchaudio>=2.11` (`pip install torchcodec`) —
> if it's missing, loading an MP3 raises a clear error telling you to
> install it or convert the file to WAV/FLAC first.


## Migration from All-In-One

All-In-One-Infer is designed to be a drop-in replacement with enhanced features. Here's how to migrate:

### Package Name Changes
```python
# Old (All-In-One)
from allin1 import analyze

# New (All-In-One-Infer)  
from allin1_infer import analyze
```

### CLI Command Changes
```shell
# Old
allin1 track.wav -o ./results

# New
all-in-one-infer track.wav -o ./results
```

### Dependency Changes
```toml
# Old dependencies (All-In-One - original)
dependencies = ["demucs", "natten>=0.15.0"]

# Current (3.0.0+) dependencies — NATTEN is no longer required
dependencies = ["torch>=2.0.0", "demucs-infer", ...]  # pure-PyTorch neighborhood attention
# Optional fused-kernel backend: pip install "all-in-one-infer[natten]"  (natten>=0.17.1,<0.20)
```

### Installation Methods

All-In-One-Infer supports both **UV** (recommended, faster) and **pip** (traditional):

```bash
# With UV (recommended, faster dependency resolution)
uv pip install git+https://github.com/openmirlab/all-in-one-infer.git

# With traditional pip (still fully supported)
pip install git+https://github.com/openmirlab/all-in-one-infer.git

# Editable install for development (works with both)
git clone https://github.com/openmirlab/all-in-one-infer.git
cd all-in-one-infer
uv pip install -e .
# or
pip install -e .
```

**Note**: The package uses modern `pyproject.toml` metadata (PEP 621 standards). All dependencies (including demucs-infer and, as of 3.1.0, madmom-infer) are installed automatically from PyPI — no GitHub git-install step needed.

### What Stays the Same
- All analysis results format (JSON structure unchanged)
- All function signatures and return types
- All model names and parameters
- All core functionality and accuracy
- All visualization and sonification features

## Training
This package is inference-only; training code has been removed (see [Scope](#scope) above — this is permanent, not a temporary gap). [TRAINING.md](docs/TRAINING.md) is kept as a historical guide. To retrain models, refer to the upstream [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) repository.

## What This Project Will Never Bundle

All-In-One-Infer downloads pretrained model checkpoints at runtime from three
different sources; it does not, and will never, ship any of them inside the
git repository or the published package.

- **No weight files in the repo or the PyPI package.** `.th`/`.pth`
  checkpoints are never committed to git and never included in the
  sdist/wheel.
- **Own `harmonix-*` models** are downloaded from the upstream author's
  Hugging Face repository, [huggingface.co/taejunkim/allinone](https://huggingface.co/taejunkim/allinone),
  via `huggingface_hub.hf_hub_download` — no re-hosted mirror.
- **Source-separation checkpoints** are downloaded by the sibling
  [demucs-infer](https://github.com/openmirlab/demucs-infer) package
  directly from Meta's official CDN — see that package's own
  ["What This Project Will Never Bundle"](https://github.com/openmirlab/demucs-infer#what-this-project-will-never-bundle)
  section.
- **Spectrogram/beat-decoding models**, if any are required, are handled
  entirely inside [madmom-infer](https://github.com/openmirlab/madmom-infer)
  — this package never downloads or bundles them directly.
- **No silently altered or re-derived weights.** This package does not
  quantize, prune, or fine-tune the upstream models and ship the result as
  a default.

All downloaded checkpoints land in `~/.cache/torch/hub/checkpoints/` (see
[Cache Management](#cache-management) above for tooling to inspect and clear
that cache).

## Development

This is a source-code-only package: see [Installation from GitHub](#installation-from-github-development)
above for an editable install. Dev-only dependencies (`black`, `ruff`) are
declared under the `dev` extra in `pyproject.toml`.

```bash
# Editable install with dev tooling
pip install -e ".[dev]"

# Run the test suite (uses the demo tracks under assets/; needs network on
# first run to download checkpoints)
pytest tests/ -v
```

See [CLAUDE.md](CLAUDE.md) for this project's file-header convention,
testing philosophy, and the exact verification commands for a change here.

## License

**MIT** (same as All-In-One and Demucs).

This package includes code from multiple sources under the MIT License:
1. [All-In-One](https://github.com/mir-aidj/all-in-one) — Copyright (c) 2023 Taejun Kim
2. Integration and modifications — Copyright (c) 2025 Bo-Yu Chen

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details and third-party dependency licenses.

## Support

- **Issues / bug reports**: [GitHub Issues](https://github.com/openmirlab/all-in-one-infer/issues)
- **Migration questions**: see [Migration from All-In-One](#migration-from-all-in-one) above
- **Version history**: see [docs/CHANGELOG.md](docs/CHANGELOG.md)
- **Upstream research questions**: see the original [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) repository

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[USAGE_EXAMPLES.md](docs/USAGE_EXAMPLES.md)** - Detailed usage examples and code snippets
- **[TRAINING.md](docs/TRAINING.md)** - Historical training guide (this package is inference-only; see note in the doc)
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history and release notes

For more information, see the [Documentation Index](docs/README.md).
