from __future__ import annotations
import logging, numpy as np, pandas as pd
from typing import Dict, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def _embed(encoder, texts: List[str], batch_size: int = 256) -> np.ndarray:
    return np.asarray(encoder.encode(texts, batch_size=batch_size,
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False))

def build_intent_descriptions(train_df: pd.DataFrame, text_col: str = "text",
                              label_col: str = "intent", top_k_terms: int = 8,
                              max_features: int = 20000, ngram_range=(1,2)) -> Dict[str,str]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
    _ = vec.fit_transform(train_df[text_col].astype(str).values)
    vocab = np.array(vec.get_feature_names_out())
    label_desc: Dict[str, str] = {}
    for lab, block in train_df.groupby(label_col):
        Xi = vec.transform(block[text_col].astype(str).values)
        mean = np.asarray(Xi.mean(axis=0)).ravel()
        if mean.sum() == 0:
            toks = " ".join(block[text_col].astype(str).tolist()).split()
            terms = [t for t,_ in pd.Series(toks).value_counts().head(top_k_terms).items()]
        else:
            idx = np.argsort(mean)[::-1][:top_k_terms]
            terms = [vocab[i] for i in idx if mean[i] > 0]
        label_desc[str(lab)] = "This intent is about: " + ", ".join(terms)
    return label_desc

def discover_superintents(train_df: pd.DataFrame, k_range: Tuple[int,int]=(8,12),
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          text_col: str = "text", label_col: str = "intent",
                          top_k_terms: int = 8, random_state: int = 42):
    logger = logging.getLogger("intentzero2few")
    df = train_df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str)
    label_desc = build_intent_descriptions(df, text_col=text_col, label_col=label_col, top_k_terms=top_k_terms)
    intents = sorted(label_desc.keys())
    enc = _get_encoder(model_name)
    embs = _embed(enc, [label_desc[i] for i in intents], batch_size=256)

    k_min, k_max = int(k_range[0]), int(k_range[1])
    k_scores = {}; best = (-1.0, None, None)
    for k in range(k_min, k_max+1):
        if k <= 1 or k >= len(intents): continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(embs)
        sil = silhouette_score(embs, labels) if 1 < k < len(intents) else -1.0
        k_scores[k] = float(sil); logger.info("discovery: K=%d silhouette=%.4f", k, sil)
        if sil > best[0]: best = (sil, k, labels)
    if best[1] is None:
        labels = np.zeros(len(intents), dtype=int); k_best = 1
    else:
        k_best = int(best[1]); labels = best[2]
    intent_to_super, super_to_intents = {}, {}
    for idx, intent in enumerate(intents):
        sid = f"S{int(labels[idx])}"
        intent_to_super[intent] = sid
        super_to_intents.setdefault(sid, []).append(intent)
    artifacts = {"model_name": model_name, "label_desc": label_desc, "intents_sorted": intents,
                 "embeddings": embs.astype(np.float32), "k_best": k_best, "k_scores": k_scores}
    return intent_to_super, super_to_intents, artifacts
