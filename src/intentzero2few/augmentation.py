"""
Text augmentation utilities.

Includes:
1) Classic EDA-style augmentation (synonyms / deletion / swap).
   - Requires NLTK WordNet (optionally).
2) Noisy augmentation (emoji / slang / character noise).
   - Explicit probabilities: p_emoji, p_slang, p_char.

EN: Use classic for semantic-preserving augmentation to grow data to a target size.
TR: Veri setini büyütmek için "anlamı koruyan" klasik augment; yanı sıra,
    robustness için "noisy" (emoji/argo/typo) augment.
"""
from __future__ import annotations
import re, random
import numpy as np
import pandas as pd

# -------------------------------
# Classic EDA-style augmentation
# -------------------------------
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
    toks = str(text).split()
    L = len(toks)
    if L == 0:
        return str(text)
    n_sr = max(1, int(alpha_sr*L))
    n_rs = max(1, int(alpha_rs*L))
    try:
        t1 = synonym_replacement(toks, n=n_sr, seed=seed)
    except RuntimeError:
        t1 = toks
    t2 = random_deletion(t1, p=alpha_rd, seed=seed)
    t3 = random_swap(t2, n=n_rs, seed=seed)
    return " ".join(t3)

def make_augmented_df(df: pd.DataFrame, per_example=1, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for _,row in df.iterrows():
        rows.append(row)
        for _ in range(per_example):
            rows.append(pd.Series({
                "text": augment_text(row["text"], seed=int(rng.randint(0,1e9))),
                "intent": row["intent"]
            }))
    return pd.DataFrame(rows).reset_index(drop=True)

# -------------------------------
# Noisy augmentation (emoji/slang/char)
# -------------------------------
_EMOJIS = ["😂","🤣","😊","😍","🔥","✨","👌","😅","😎","🤔","🙃","💯","🤷","😩","🥲","🤗","🙌","😭","😴","😜"]
_SLANG  = {
  "you":"u", "are":"r", "your":"ur", "please":"pls", "people":"ppl",
  "thanks":"thx", "thank you":"ty", "because":"cuz", "okay":"ok",
  "really":"rly", "message":"msg", "before":"b4", "tomorrow":"tmrw",
  "between":"btwn", "favorite":"fav", "see you":"cu", "by the way":"btw",
  "for your information":"fyi", "as soon as possible":"asap", "I don't know":"idk"
}

def _inject_emojis(text:str, rng:np.random.RandomState, min_n=1, max_n=3)->str:
    n = int(rng.randint(min_n, max_n+1))
    em = "".join(rng.choice(_EMOJIS, size=n))
    # 50% append, 50% inline
    if rng.rand() < 0.5:
        return text.strip() + " " + em
    toks = text.split()
    if not toks:
        return em
    pos = int(rng.randint(0, len(toks)))
    toks.insert(pos, em)
    return " ".join(toks)

def _slangify(text:str, rng:np.random.RandomState)->str:
    s = " " + text.lower() + " "
    # longest keys first to avoid partial overlaps
    for k in sorted(_SLANG.keys(), key=len, reverse=True):
        if rng.rand() < 0.5 and f" {k} " in s:
            s = s.replace(f" {k} ", f" {_SLANG[k]} ")
    # random add-ons
    tails = [" lol", " lmao", " smh", " fr", " tbh", " ngl", " btw", " idk"]
    if rng.rand() < 0.3:
        s = s.strip() + rng.choice(tails)
    return s.strip()

def _char_noise(text:str, rng:np.random.RandomState, p_char:float=0.05)->str:
    # light character-level noise: swap/duplicate/delete/case flip
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if rng.rand() < p_char and ch.isalpha():
            op = rng.choice(["dup","del","swap","case"])
            if op == "dup":
                out.append(ch); out.append(ch)
            elif op == "del":
                # skip this char (delete)
                i += 1
                continue
            elif op == "swap" and i+1 < len(text):
                out.append(text[i+1]); out.append(ch)
                i += 2
                continue
            elif op == "case":
                out.append(ch.upper() if ch.islower() else ch.lower())
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)

def make_noisy_text(text:str,
                    rng:np.random.RandomState,
                    p_emoji:float=0.3, p_slang:float=0.3, p_char:float=0.05)->str:
    s = str(text)
    if rng.rand() < p_slang:
        s = _slangify(s, rng)
    if rng.rand() < p_emoji:
        s = _inject_emojis(s, rng)
    if p_char > 0:
        s = _char_noise(s, rng, p_char=p_char)
    return s

def make_noisy_df(df: pd.DataFrame,
                  per_example:int=1, seed:int=42,
                  p_emoji:float=0.3, p_slang:float=0.3, p_char:float=0.05) -> pd.DataFrame:
    """
    EN: Return df with original rows + `per_example` noisy variants per row.
    TR: Her satıra `per_example` adet gürültülü kopya ekler (orijinali korur).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for _,row in df.iterrows():
        rows.append(row)
        for _ in range(per_example):
            rows.append(pd.Series({
                "text": make_noisy_text(row["text"], rng, p_emoji=p_emoji, p_slang=p_slang, p_char=p_char),
                "intent": row["intent"]
            }))
    return pd.DataFrame(rows).reset_index(drop=True)
