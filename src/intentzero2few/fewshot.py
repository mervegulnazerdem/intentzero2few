from __future__ import annotations
import numpy as np, pandas as pd

def make_k_shot(train_df: pd.DataFrame, k: int = 5, seed: int = 42, drop_short: bool = False) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    out = []
    for intent, g in train_df.groupby("intent"):
        if len(g) < k:
            if drop_short: continue
            out.append(g)
        else:
            out.append(g.sample(n=k, random_state=rng))
    if not out: raise ValueError("No class has >=k samples.")
    return pd.concat(out).reset_index(drop=True)
