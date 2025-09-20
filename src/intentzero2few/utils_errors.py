from __future__ import annotations
import os, pandas as pd
from .utils_io import run_path

def error_csv_path(component: str = "generic") -> str:
    return run_path("analytics", f"errors_{component}.csv")

def log_error_row(component: str = "generic", **fields):
    path = error_csv_path(component)
    row = pd.DataFrame([fields])
    if os.path.exists(path):
        old = pd.read_csv(path)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)
    return path
