from __future__ import annotations
import logging, numpy as np, pandas as pd
from typing import Dict, List

def _get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def _embed(encoder, texts: List[str], batch_size: int = 256) -> np.ndarray:
    return np.asarray(encoder.encode(texts, batch_size=batch_size,
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False))

class ZeroShotSuperIntent:
    def __init__(self, model_name: str, super_labels: List[str],
                 centroids: np.ndarray, intent_to_super: Dict[str,str]):
        self.model_name = model_name
        self.super_labels = list(super_labels)
        self.centroids = np.asarray(centroids, np.float32)
        self.intent_to_super = dict(intent_to_super)
        self._encoder = None
    def _ensure(self):
        if self._encoder is None:
            self._encoder = _get_encoder(self.model_name)
    def predict_proba(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        self._ensure(); import numpy as np
        X = _embed(self._encoder, list(map(str, texts)), batch_size=batch_size)
        return np.clip(X @ self.centroids.T, -1.0, 1.0)
    def predict(self, texts: List[str], tau: float, batch_size: int = 256) -> List[str]:
        S = self.predict_proba(texts, batch_size=batch_size)
        idx = S.argmax(axis=1); smax = S.max(axis=1)
        return [(self.super_labels[int(i)] if s >= float(tau) else "OOS") for s, i in zip(smax, idx)]

def fit_superintent_zeroshot(train_df: pd.DataFrame, intent_to_super: Dict[str,str], artifacts: Dict,
                             exemplars_per_intent: int = 5, description_weight: float = 1.0,
                             model_name: str | None = None, text_col: str = "text", label_col: str = "intent",
                             random_state: int = 42) -> ZeroShotSuperIntent:
    logger = logging.getLogger("intentzero2few")
    if model_name is None:
        model_name = artifacts.get("model_name","sentence-transformers/all-MiniLM-L6-v2")
    df = train_df[[text_col, label_col]].dropna().copy(); df[label_col] = df[label_col].astype(str)
    label_desc = artifacts["label_desc"]
    super_to_intents: Dict[str, List[str]] = {}
    for intent, s in intent_to_super.items():
        super_to_intents.setdefault(s, []).append(intent)
    super_labels = sorted(super_to_intents.keys())
    enc = _get_encoder(model_name); import numpy as np; rng = np.random.RandomState(random_state)
    centroids = []
    for s in super_labels:
        members = super_to_intents[s]
        desc_texts = [label_desc[i] for i in members]
        desc_emb = np.asarray(enc.encode(desc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        if description_weight != 1.0: desc_emb = desc_emb * float(description_weight)
        ex_texts: List[str] = []
        for intent in members:
            block = df[df[label_col] == intent]
            k = min(len(block), int(exemplars_per_intent))
            if k > 0: ex_texts.extend(block.sample(n=k, random_state=rng)[text_col].astype(str).tolist())
        ex_emb = (np.asarray(enc.encode(ex_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
                  if ex_texts else np.zeros((0, desc_emb.shape[1]), np.float32))
        all_emb = desc_emb if len(ex_emb) == 0 else (ex_emb if len(desc_emb) == 0 else np.vstack([desc_emb, ex_emb]))
        c = all_emb.mean(axis=0); norm = np.linalg.norm(c) + 1e-12; centroids.append(c / norm)
        logger.info("zeroshot: centroid %s built with %d desc + %d exemplars", s, len(desc_emb), len(ex_emb))
    centroids = np.vstack(centroids).astype(np.float32)
    return ZeroShotSuperIntent(model_name=model_name, super_labels=super_labels, centroids=centroids, intent_to_super=intent_to_super)
