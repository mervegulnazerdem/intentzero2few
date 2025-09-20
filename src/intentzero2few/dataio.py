from __future__ import annotations
from typing import Dict, Optional
import json, pandas as pd

def load_intents(path: str) -> Dict[str, pd.DataFrame]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    def to_df(key: str) -> Optional[pd.DataFrame]:
        return pd.DataFrame(data[key], columns=["text","intent"]) if key in data else None
    splits = {"train":to_df("train"), "val":to_df("val"), "test":to_df("test")}
    for req in ["train","val","test"]:
        if splits[req] is None:
            raise ValueError(f"Missing split: {req}")
    for opt in ["oos_val","oos_test"]:
        df = to_df(opt)
        if df is not None:
            splits[opt] = df
    return splits
