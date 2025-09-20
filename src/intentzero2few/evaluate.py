from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
_OOS = {"oos","__neg__","out_of_scope","out-of-scope","unknown"}

def _is_oos(x: str) -> bool:
    if x is None: return True
    return str(x).strip().casefold() in _OOS

def _true_supers(df: pd.DataFrame, mapping: Dict[str,str], intent_col: str = "intent", is_oos_col: str = "is_oos") -> List[str]:
    y = []
    for _, r in df.iterrows():
        if is_oos_col in df.columns and int(r.get(is_oos_col, 0)) == 1:
            y.append("OOS"); continue
        it = str(r[intent_col]); y.append("OOS" if _is_oos(it) else mapping.get(it, "OOS"))
    return y

def evaluate_superintent(zs_model, df: pd.DataFrame, tau: float, intent_col: str = "intent", is_oos_col: str = "is_oos") -> Dict:
    texts = df["text"].astype(str).tolist()
    y_true = _true_supers(df, zs_model.intent_to_super, intent_col, is_oos_col)
    S = zs_model.predict_proba(texts); idx = S.argmax(axis=1); smax = S.max(axis=1)
    y_pred = ["OOS" if s < tau else zs_model.super_labels[int(i)] for s, i in zip(smax, idx)]
    labels = sorted(set(zs_model.super_labels) | {"OOS"})
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True); cmn = np.nan_to_num(cmn)
    acc = accuracy_score(y_true, y_pred); macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return {"labels": labels, "classification_report": report,
            "confusion_matrix": cm.tolist(), "confusion_matrix_normalized": cmn.tolist(),
            "accuracy": float(acc), "macro_f1": float(macro_f1)}
