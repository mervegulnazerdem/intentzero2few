
"""
Zero-shot super-intent classifier
Goal: Build one prototype (centroid) per discovered super-intent, then score
new texts by cosine similarity (via dot product on L2-normalized embeddings).
Steps:
  (1) Group fine-grained intents by their super-intent (from discovery).
  (2) Encode per-intent description texts → description embeddings.
  (3) Sample up to K exemplar utterances per member intent from TRAIN and encode.
  (4) Centroid = mean(description_embs ∪ exemplar_embs), then L2-normalize.
  (5) Inference: S = X · C^T  (cosine), predict argmax if max_sim ≥ τ, else OOS."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

# -- Lazy model init
def _get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def _embed(encoder, texts: List[str], batch_size: int = 256) -> np.ndarray:

    """Encode and L2-normalize."""

    return np.asarray(
        encoder.encode(
            texts, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
    )

class ZeroShotSuperIntent:

    """Holds super labels, centroids and the encoder; provides predict(_proba)."""

    def __init__(self, model_name: str, super_labels: List[str], centroids: np.ndarray, intent_to_super: Dict[str,str]):
        self.model_name = model_name
        self.super_labels = list(super_labels)                 # e.g. ["S0","S1",...]
        self.centroids = np.asarray(centroids, np.float32)     # (K, D), L2-normalized
        self.intent_to_super = dict(intent_to_super)
        self._encoder = None

    def _ensure(self):
        if self._encoder is None:
            self._encoder = _get_encoder(self.model_name)

    def predict_proba(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        
        """Return similarity matrix S \in R^{N x K}.With normalized vectors, cosine == dot product."""
        
        self._ensure()
        X = _embed(self._encoder, list(map(str, texts)), batch_size=batch_size)
        return np.clip(X @ self.centroids.T, -1.0, 1.0)

    def predict(self, texts: List[str], tau: float, batch_size: int = 256) -> List[str]:

        """Thresholded prediction: argmax if max_sim >= tau, otherwise 'OOS'."""

        S = self.predict_proba(texts, batch_size=batch_size)
        idx = S.argmax(axis=1)
        smax = S.max(axis=1)
        return [(self.super_labels[int(i)] if s >= float(tau) else "OOS") for s, i in zip(smax, idx)]

def fit_superintent_zeroshot(
    train_df: pd.DataFrame,
    intent_to_super: Dict[str,str],
    artifacts: Dict,
    exemplars_per_intent: int = 5,
    description_weight: float = 1.0,
    model_name: str | None = None,
    text_col: str = "text",
    label_col: str = "intent",
    random_state: int = 42,
) -> ZeroShotSuperIntent:
    
    """Build super-intent centroids using both description and exemplar embeddings."""
    
    logger = logging.getLogger("intentzero2few")
    if model_name is None:
        model_name = artifacts.get("model_name","sentence-transformers/all-MiniLM-L6-v2")

    df = train_df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str)

    # (1) Map member intents per super-intent
    label_desc = artifacts["label_desc"]
    super_to_intents: Dict[str, List[str]] = {}
    for intent, s in intent_to_super.items():
        super_to_intents.setdefault(s, []).append(intent)
    super_labels = sorted(super_to_intents.keys())

    enc = _get_encoder(model_name)
    rng = np.random.RandomState(random_state)
    centroids = []

    for s in super_labels:
        members = super_to_intents[s]

        # (2) Description embeddings (one per member intent)
        desc_texts = [label_desc[i] for i in members]
        desc_emb = _embed(enc, desc_texts, 256)
        if description_weight != 1.0:
            desc_emb = desc_emb * float(description_weight)

        # (3) Exemplar embeddings (sampled from TRAIN)
        ex_texts: List[str] = []
        for intent in members:
            block = df[df[label_col] == intent]
            k = min(len(block), int(exemplars_per_intent))
            if k > 0:
                ex_texts.extend(block.sample(n=k, random_state=rng)[text_col].astype(str).tolist())
        ex_emb = _embed(enc, ex_texts, 256) if ex_texts else np.zeros((0, desc_emb.shape[1]), np.float32)

        # (4) Centroid = mean, then L2-normalize
        all_emb = desc_emb if len(ex_emb) == 0 else (ex_emb if len(desc_emb) == 0 else np.vstack([desc_emb, ex_emb]))
        c = all_emb.mean(axis=0)
        norm = np.linalg.norm(c) + 1e-12
        centroids.append(c / norm)

        logger.info("zeroshot: built centroid for %s with %d desc + %d exemplars", s, len(desc_emb), len(ex_emb))

    centroids = np.vstack(centroids).astype(np.float32)
    return ZeroShotSuperIntent(model_name=model_name, super_labels=super_labels, centroids=centroids, intent_to_super=intent_to_super)
