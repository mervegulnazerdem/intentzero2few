from .core import set_all_seeds, SEED, TEXT_COL, LABEL_COL
from .dataio import load_intents
from .eda import quick_eda, plot_top_intents, sample_by_intent, show_split
from .fewshot import make_k_shot
from .pollution import generate_fallback_negatives_en, make_polluted_test, make_polluted_test_debug
from .labeling import fit_label_encoder, encode_in_scope_labels, sanity_check_labels
from .baselines import (
    split_in_scope, eval_in_scope, select_threshold_msp, oos_metrics,
    TfidfLR, TfidfLinearSVM, BertLinear, MajorityClassifier
)
from .discovery import discover_superintents, build_intent_descriptions
from .zeroshot import fit_superintent_zeroshot
from .threshold import calibrate_threshold
from .evaluate import evaluate_superintent
from .utils_logging import setup_logger
from .utils_io import get_env_paths, run_path, report_path, save_json, save_csv, save_figure, copy_to_report
from .utils_errors import log_error_row, error_csv_path

# --- Safe, fresh submodules (avoid stale cache in Colab) ---
import sys as _sys, importlib as _importlib
augmentation = None
try:
    if "intentzero2few.augmentation" in _sys.modules:
        _importlib.reload(_sys.modules["intentzero2few.augmentation"])
    from . import augmentation as augmentation
except Exception:
    augmentation = None

viz = None
try:
    if "intentzero2few.viz" in _sys.modules:
        _importlib.reload(_sys.modules["intentzero2few.viz"])
    from . import viz as viz
except Exception:
    viz = None
