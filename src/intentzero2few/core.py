from __future__ import annotations
import random, numpy as np
SEED = 42
TEXT_COL = "text"
LABEL_COL = "intent"

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
