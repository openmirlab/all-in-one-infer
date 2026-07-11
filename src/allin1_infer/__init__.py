__version__ = "2.0.0"

# Check for required madmom dependency
# Note: madmom should be auto-installed during package installation via setup.py hook
try:
    import madmom
except ImportError:
    raise ImportError(
        "madmom is required but not installed. "
        "Please install it with: pip install git+https://github.com/CPJKU/madmom\n"
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
from .helpers import (
    get_model_cache_dir,
    get_cache_size,
    list_cached_models,
    clear_model_cache,
    print_cache_info
)
