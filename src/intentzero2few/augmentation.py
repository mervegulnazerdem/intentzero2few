import re
import numpy as np
import pandas as pd

def _safe_import_wordnet():
    try:
        from nltk.corpus import wordnet
        return wordnet
    except Exception as e:
        raise RuntimeError("Run once: import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')") from e

def get_synonyms(word:str):
    wn = _safe_import_wordnet()
    syns = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            w = l.name().replace("_"," ")
            if w.lower() != word.lower():
                syns.add(w)
    return list(syns)

def synonym_replacement(tokens, n=1, seed=42):
    rng = np.random.RandomState(seed)
    tokens = tokens.copy()
    cand = [w for w in tokens if re.match(r"^[A-Za-z]+$", w)]
    rng.shuffle(cand)
    cnt = 0
    for w in cand:
        syns = get_synonyms(w)
        if syns:
            rep = rng.choice(syns)
            idx = tokens.index(w)
            tokens[idx] = rep
            cnt += 1
            if cnt >= n:
                break
    return tokens

def random_deletion(tokens, p=0.1, seed=42):
    rng = np.random.RandomState(seed)
    if len(tokens) == 1:
        return tokens
    keep = [t for t in tokens if rng.rand() > p]
    return keep if keep else tokens

def random_swap(tokens, n=1, seed=42):
    rng = np.random.RandomState(seed)
    tokens = tokens.copy()
    for _ in range(n):
        if len(tokens) < 2:
            break
        i, j = rng.choice(range(len(tokens)), 2, replace=False)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens

def augment_text(text, alpha_sr=0.1, alpha_rd=0.1, alpha_rs=0.1, seed=42):
    toks = text.split()
    L = len(toks)
    if L == 0:
        return text
    n_sr = max(1, int(alpha_sr*L))
    n_rs = max(1, int(alpha_rs*L))
    try:
        t1 = synonym_replacement(toks, n=n_sr, seed=seed)
    except RuntimeError:
        t1 = toks
    t2 = random_deletion(t1, p=alpha_rd, seed=seed)
    t3 = random_swap(t2, n=n_rs, seed=seed)
    return " ".join(t3)

def make_augmented_df(df, per_example=1, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for _,row in df.iterrows():
        rows.append(row)
        for _ in range(per_example):
            rows.append(pd.Series({"text":augment_text(row["text"], seed=int(rng.randint(0,1e9))), "intent":row["intent"]}))
    return pd.DataFrame(rows).reset_index(drop=True)
