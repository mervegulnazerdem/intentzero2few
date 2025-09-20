from __future__ import annotations
import numpy as np, pandas as pd
from typing import Iterable, Dict, List
from sklearn.metrics import f1_score
_OOS = {"oos","__neg__","out_of_scope","out-of-scope","unknown"}

def _is_oos(x:str)->bool:
    if x is None: return True
    return str(x).strip().casefold() in _OOS

def _true_supers(df: pd.DataFrame, mapping: Dict[str,str], intent_col="intent", is_oos_col="is_oos")->List[str]:
    y = []
    for _, r in df.iterrows():
        if is_oos_col in df.columns and int(r.get(is_oos_col, 0)) == 1:
            y.append("OOS"); continue
        it = str(r[intent_col])
        y.append("OOS" if _is_oos(it) else mapping.get(it, "OOS"))
    return y

def calibrate_threshold(zs_model, val_df: pd.DataFrame, tau_grid: Iterable[float] | None = None,
                        intent_col="intent", is_oos_col="is_oos")->float:
    if tau_grid is None: tau_grid = np.linspace(0.2,0.9,36)
    texts = val_df["text"].astype(str).tolist()
    y_true = _true_supers(val_df, zs_model.intent_to_super, intent_col, is_oos_col)
    S = zs_model.predict_proba(texts); idx = S.argmax(axis=1); smax = S.max(axis=1)
    labels = sorted(set(zs_model.super_labels) | {"OOS"})
    best_tau,best_f1 = None,-1.0
    for tau in tau_grid:
        y_pred = ["OOS" if s<float(tau) else zs_model.super_labels[int(i)] for s,i in zip(smax, idx)]
        f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        if (f1>best_f1) or (np.isclose(f1,best_f1) and (best_tau is None or tau>best_tau)):
            best_tau,best_f1 = float(tau),float(f1)
    return float(best_tau)
