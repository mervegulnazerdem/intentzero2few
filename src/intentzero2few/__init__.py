from .core import set_all_seeds, SEED, TEXT_COL, LABEL_COL
from .dataio import load_intents
from .eda import quick_eda, plot_top_intents, sample_by_intent, show_split
from .fewshot import make_k_shot
from .pollution import generate_fallback_negatives_en, make_polluted_test, make_polluted_test_debug
from .augmentation import make_augmented_df, augment_text
from .labeling import fit_label_encoder, encode_in_scope_labels, sanity_check_labels
from .baselines import (
    run_majority, run_tfidf_lr, run_bert_linear,
    split_in_scope, eval_in_scope, select_threshold_msp, oos_metrics,
    TfidfLR, TfidfLinearSVM, BertLinear, MajorityClassifier
)
from .discovery import discover_superintents
from .zeroshot import fit_superintent_zeroshot
from .threshold import calibrate_threshold
from .evaluate import evaluate_superintent
from .utils_logging import setup_logger
