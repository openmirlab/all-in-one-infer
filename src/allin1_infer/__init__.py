"""Public API surface of the allin1_infer package.

Re-exports the user-facing entry points (analyze, visualize, sonify, stem
providers, cache helpers) so `import allin1_infer` gives a flat API without
callers needing to know the internal module layout. Hard-fails at import time
if madmom_infer isn't installed, since madmom_infer's DBN beat/downbeat
decoder (postprocessing/metrical.py) is a non-optional part of the inference
pipeline, not a soft dependency.

Reads: .analyze, .visualize, .sonify, .stems, .stems_input, .cache, .config, .utils, .__about__
"""

from .__about__ import __version__

# Check for required madmom_infer dependency
# Note: madmom_infer is a plain PyPI dependency, installed automatically
# alongside allin1_infer -- no post-install hook needed.
try:
    import madmom_infer
except ImportError:
    raise ImportError(
        "madmom_infer is required but not installed. "
        "Please install it with: pip install madmom-infer\n"
        "If you just installed allin1_infer, the auto-install may have failed.\n"
        "See README.md for complete installation instructions."
    )

from .analyze import analyze
from .visualize import visualize
from .sonify import sonify
from .typings import AnalysisResult
from .config import HARMONIX_LABELS
from .utils import load_result
from .stems import (
    StemProvider,
    DemucsProvider,
    PrecomputedStemProvider,
    CustomSeparatorProvider,
    get_stems
)
from .stems_input import (
    StemsInput,
    create_stems_input_from_directory,
    create_stems_input_from_pattern,
    prepare_stems_for_analysis
)
from .cache import (
    get_model_cache_dir,
    get_cache_size,
    list_cached_models,
    clear_model_cache,
    print_cache_info
)
