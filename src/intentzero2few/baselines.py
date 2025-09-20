from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def split_in_scope(df_enc: pd.DataFrame, text_col="text", y_col="label_id"):
    mask = df_enc[y_col].astype(int) >= 0
    X = df_enc.loc[mask, text_col].astype(str).tolist()
    y = df_enc.loc[mask, y_col].astype(int).values
    return X, y

def eval_in_scope(df_enc: pd.DataFrame, y_pred: np.ndarray, y_col="label_id"):
    y_true = df_enc.loc[df_enc[y_col] >= 0, y_col].astype(int).values
    return {"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro"))}

def select_threshold_msp(polluted_val_df_enc: pd.DataFrame, msp_val: np.ndarray,
                         oos_label_id: int = -1, target_tpr: float = 0.95):
    is_oos = (polluted_val_df_enc["label_id"].astype(int)==oos_label_id).astype(int).values
    det = -msp_val
    auroc = roc_auc_score(is_oos, det)
    fpr, tpr, thr = roc_curve(is_oos, det)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return {"tau":float(-thr[idx]), "auroc":float(auroc), "fpr_at_tpr":float(fpr[idx]), "target_tpr":float(tpr[idx])}

def oos_metrics(polluted_test_df_enc: pd.DataFrame, msp_test: np.ndarray, tau: float,
                oos_label_id: int = -1, y_pred_in_scope: Optional[np.ndarray] = None):
    is_oos = (polluted_test_df_enc["label_id"].astype(int)==oos_label_id).astype(int).values
    det = -msp_test
    fpr, tpr, thr = roc_curve(is_oos, det)
    auroc = roc_auc_score(is_oos, det)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr95 = float(fpr[idx])
    out = {"auroc_oos":float(auroc), "fpr@tpr95":fpr95, "tau_used":float(tau)}
    if y_pred_in_scope is not None:
        accept = (msp_test>=tau) & (is_oos==0)
        y_true = polluted_test_df_enc.loc[accept,"label_id"].astype(int).values
        y_pred = y_pred_in_scope[accept]
        from sklearn.metrics import accuracy_score
        out["in_scope_acc_on_accepted"] = float(accuracy_score(y_true,y_pred)) if len(y_true)>0 else float("nan")
    return out

class MajorityClassifier:
    def fit(self, y):
        vals,counts = np.unique(y, return_counts=True)
        self.major = int(vals[np.argmax(counts)]); return self
    def predict(self, X): return np.full((len(X),), self.major, dtype=int)
    def predict_proba(self, X, n_classes:int):
        p = np.zeros((len(X), n_classes), float); p[:,self.major] = 1.0; return p

class TfidfLR:
    def __init__(self,max_features=30000,ngram_range=(1,2),min_df=2,C=1.0):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        self.n_classes_ = None
    def fit(self,X_train,y_train):
        Xtr = self.vec.fit_transform(X_train); self.clf.fit(Xtr,y_train)
        self.n_classes_ = int(self.clf.classes_.shape[0]); return self
    def predict(self,X):
        Xte = self.vec.transform(X); return self.clf.predict(Xte)
    def predict_proba(self,X):
        Xte = self.vec.transform(X); return self.clf.predict_proba(Xte)

class TfidfLinearSVM:
    def __init__(self,max_features=30000,ngram_range=(1,2),min_df=2,C=1.0):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.clf = LinearSVC(C=C); self.n_classes_ = None
    def fit(self,X,y):
        Xtr = self.vec.fit_transform(X); self.clf.fit(Xtr,y); self.n_classes_ = len(np.unique(y)); return self
    def predict(self,X):
        Xte = self.vec.transform(X); return self.clf.predict(Xte)
    def msp_like(self,X):
        Xte = self.vec.transform(X)
        margins = self.clf.decision_function(Xte)
        if margins.ndim==1:
            import numpy as _np
            margins = _np.vstack([-margins,margins]).T
        m = margins - margins.max(axis=1,keepdims=True)
        e = np.exp(m); p = e / e.sum(axis=1,keepdims=True)
        return p.max(axis=1)

def _safe_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None

class BertLinear:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.SentenceTransformer = _safe_st()
        self.model = None
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.dim_ = None; self.n_classes_ = None
    def _ensure(self):
        if self.SentenceTransformer is None:
            raise ImportError("Install sentence-transformers")
        if self.model is None:
            self.model = self.SentenceTransformer(self.model_name)
    def _embed(self,texts:list[str]):
        self._ensure()
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    def fit(self,X,y):
        Xemb = self._embed(X); self.dim_ = Xemb.shape[1]; self.clf.fit(Xemb,y)
        self.n_classes_ = int(self.clf.classes_.shape[0]); return self
    def predict(self,X): return self.clf.predict(self._embed(X))
    def predict_proba(self,X): return self.clf.predict_proba(self._embed(X))
