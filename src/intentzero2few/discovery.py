
"""Super-intent discovery
Goal: Group fine-grained intents into higher-level super-intents automatically.

Pipeline:
  (1) Build per-intent text descriptions using top TF-IDF terms.
  (2) Embed those descriptions with Sentence-Transformers (L2-normalized).
  (3) Choose K in a given range [k_min, k_max] via silhouette score and cluster with KMeans.
  (4) Produce mappings:
        - intent_to_super[intent] = "S{cluster_id}"
        - super_to_intents["S{cluster_id}"] = [intent_1, intent_2, ...]
      and return helpful artifacts (descriptions, embeddings, K scores)."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -- Lazy encoder import keeps import-time deps light.
def _get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def _embed(encoder, texts: List[str], batch_size: int = 256) -> np.ndarray:
    """Encode text list -> L2-normalized embeddings (np.ndarray)."""
    return np.asarray(
        encoder.encode(
            texts, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
    )

def build_intent_descriptions(
    train_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "intent",
    top_k_terms: int = 8,
    max_features: int = 20000,
    ngram_range=(1,2),
) -> Dict[str,str]:

    """(1) Build per-intent descriptions:
        - Fit TF-IDF on the whole TRAIN.
        - For each intent, compute mean TF-IDF across its samples.
        - Take top-K terms; fallback to raw token frequencies if mean TF-IDF is all zeros.
        - Return a short, human-friendly sentence per intent."""

    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
    _ = vec.fit_transform(train_df[text_col].astype(str).values)
    vocab = np.array(vec.get_feature_names_out())

    label_desc: Dict[str, str] = {}
    for lab, block in train_df.groupby(label_col):
        # mean TF-IDF within this intent
        Xi = vec.transform(block[text_col].astype(str).values)
        mean = np.asarray(Xi.mean(axis=0)).ravel()

        if mean.sum() == 0:
            # Fallback: most frequent tokens in the block
            toks = " ".join(block[text_col].astype(str).tolist()).split()
            terms = [t for t,_ in pd.Series(toks).value_counts().head(top_k_terms).items()]
        else:
            idx = np.argsort(mean)[::-1][:top_k_terms]
            terms = [vocab[i] for i in idx if mean[i] > 0]

        label_desc[str(lab)] = "This intent is about: " + ", ".join(terms)
    return label_desc

def discover_superintents(
    train_df: pd.DataFrame,
    k_range: Tuple[int,int]=(8,12),
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    text_col: str = "text",
    label_col: str = "intent",
    top_k_terms: int = 8,
    random_state: int = 42,
):
    
    """
    (2) Embed intent descriptions  → embs (N_intents, D)
    (3) K selection via silhouette → pick best K and cluster with KMeans
    (4) Build mappings and artifacts for downstream zero-shot layer."""

    logger = logging.getLogger("intentzero2few")

    # Keep only the needed columns and enforce string labels
    df = train_df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str)

    # (1) Descriptions
    label_desc = build_intent_descriptions(
        df, text_col=text_col, label_col=label_col, top_k_terms=top_k_terms
    )

    # (2) Embeddings for all intent descriptions
    intents = sorted(label_desc.keys())
    enc = _get_encoder(model_name)
    embs = _embed(enc, [label_desc[i] for i in intents], batch_size=256)

    # (3) Choose K in [k_min, k_max] by silhouette
    k_min, k_max = int(k_range[0]), int(k_range[1])
    k_scores = {}
    best = (-1.0, None, None)   # (silhouette, k, labels)
    for k in range(k_min, k_max+1):
        if k <= 1 or k >= len(intents):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(embs)
        sil = silhouette_score(embs, labels) if 1 < k < len(intents) else -1.0
        k_scores[k] = float(sil)
        logger.info("discovery: K=%d silhouette=%.4f", k, sil)
        if sil > best[0]:
            best = (sil, k, labels)

    if best[1] is None:
        # Degenerate fallback: 1 cluster for all intents
        labels = np.zeros(len(intents), dtype=int)
        k_best = 1
    else:
        k_best = int(best[1])
        labels = best[2]

    # (4) Mappings
    intent_to_super: Dict[str, str] = {}
    super_to_intents: Dict[str, List[str]] = {}
    for idx, intent in enumerate(intents):
        sid = f"S{int(labels[idx])}"
        intent_to_super[intent] = sid
        super_to_intents.setdefault(sid, []).append(intent)

    artifacts = {
        "model_name": model_name,
        "label_desc": label_desc,
        "intents_sorted": intents,
        "embeddings": embs.astype(np.float32),
        "k_best": k_best,
        "k_scores": k_scores,
    }
    return intent_to_super, super_to_intents, artifacts
