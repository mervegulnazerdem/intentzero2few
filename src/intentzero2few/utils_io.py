from __future__ import annotations
import os, json
from typing import Any, Optional
import pandas as pd

def _env(name: str, default: Optional[str] = None) -> str:
    return os.environ.get(name, default or "")

def get_env_paths() -> dict:
    repo = _env("REPO_DIR", "/content/intentzero2few-repo")
    run_id = _env("RUN_ID")
    run_dir = _env("RUN_DIR") or (os.path.join(repo, "runs", run_id) if run_id else os.path.join(repo, "runs", "adhoc"))
    report_dir = _env("REPORT_DIR") or (os.path.join(repo, "reports", run_id) if run_id else os.path.join(repo, "reports", "adhoc"))
    for sub in ("analytics", "logs", "figures", "artifacts"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    return {"REPO_DIR": repo, "RUN_ID": run_id, "RUN_DIR": run_dir, "REPORT_DIR": report_dir}

def run_path(subdir: str, filename: str) -> str:
    p = get_env_paths()
    base = os.path.join(p["RUN_DIR"], subdir)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

def report_path(filename: str) -> str:
    p = get_env_paths()
    base = p["REPORT_DIR"]
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

def save_json(obj: Any, path: str, ensure_ascii: bool = False, indent: int = 2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)

def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)

def save_figure(fig, path: str, dpi: int = 300, bbox_inches: str = "tight"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

def copy_to_report(src_path: str, dst_name: Optional[str] = None) -> str:
    import shutil
    p = get_env_paths()
    dst = os.path.join(p["REPORT_DIR"], dst_name or os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst
