from __future__ import annotations
from typing import Iterable, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def _normalize_label(x)->str: return "" if x is None else str(x).strip()
def _normcase(x)->str: return _normalize_label(x).casefold()
def _oos_norm_set(oos_labels: Optional[Iterable[str]]): return { _normcase(lbl) for lbl in (oos_labels or []) }

def fit_label_encoder(train_df: pd.DataFrame, label_col: str = "intent",
                      oos_labels: Optional[Iterable[str]]=("OOS","__NEG__")):
    oos = _oos_norm_set(oos_labels)
    labs = train_df[label_col].apply(_normalize_label)
    mask = labs.apply(lambda v: _normcase(v) in oos)
    in_scope = labs[~mask]
    le = LabelEncoder().fit(in_scope.values)
    l2i = {lbl:int(i) for i,lbl in enumerate(le.classes_)}
    i2l = {int(i):lbl for lbl,i in l2i.items()}
    return le,l2i,i2l

def encode_in_scope_labels(df: pd.DataFrame, le: LabelEncoder, label_col="intent",
                           oos_labels=("OOS","__NEG__"), oos_sentinel=-1, out_col="label_id"):
    df = df.copy()
    oos = _oos_norm_set(oos_labels)
    labs = df[label_col].apply(_normalize_label)
    mask_in = ~labs.apply(lambda v: _normcase(v) in oos)
    df.loc[mask_in, out_col] = le.transform(labs[mask_in].values)
    df.loc[~mask_in, out_col] = oos_sentinel
    df[out_col] = df[out_col].astype(int)
    return df

def sanity_check_labels(df_list, out_col="label_id"):
    for i,df in enumerate(df_list, start=1):
        assert out_col in df.columns, f"missing {out_col} in df#{i}"
        assert pd.api.types.is_integer_dtype(df[out_col]), f"{out_col} must be int"
