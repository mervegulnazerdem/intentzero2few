from __future__ import annotations
import logging, os, time

def setup_logger(name: str = "intentzero2few", log_dir: str | None = None):
    if log_dir is None:
        run_dir = os.environ.get("RUN_DIR", "")
        log_dir = (os.path.join(run_dir, "logs") if run_dir else os.path.join("runs","logs"))
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_dir, f"{name}-{ts}.log")

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fh = logging.FileHandler(path, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch)
    lg.info("Logger started. path=%s", path)
    return lg, path
