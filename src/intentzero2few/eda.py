from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

def quick_eda(df: pd.DataFrame, title: str = "Dataset"):
    print(f"== {title} ==")
    print("Rows:", len(df))
    print("Unique intents:", df['intent'].nunique())
    lengths = df['text'].str.split().apply(len)
    print("Avg words:", round(lengths.mean(), 4))
    print("Min/Max:", lengths.min(), "/", lengths.max())
    print("\nTop 10 intents:\n", df['intent'].value_counts().head(10))
    return lengths

def plot_top_intents(df: pd.DataFrame, top_n: int = 20):
    c = df['intent'].value_counts()
    plt.figure(figsize=(12,6))
    sns.barplot(x=c[:top_n].values, y=c[:top_n].index)
    plt.title(f'Top {top_n} Intents')
    plt.tight_layout(); plt.show()

def sample_by_intent(df, n_intents=6, n_per_intent=1, label_col="intent", text_col="text", random_state=42):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    if label_col not in df.columns:
        df[label_col] = "oos"
    shown = [c for c in [label_col, text_col] if c in df.columns] or df.columns[:2].tolist()
    uniq = df[label_col].dropna().astype(str).unique().tolist()
    lower = [u.lower() for u in uniq]
    single = (len(uniq)==1 and lower[0] in {"oos","out_of_scope","out-of-scope","unknown"})
    if single or len(uniq)==1:
        k = min(len(df), n_intents*n_per_intent)
        if k <= 0: return df.iloc[:0,:][shown]
        idx = rng.choice(len(df), size=k, replace=False)
        return df.iloc[idx][shown].reset_index(drop=True)
    import numpy as _np
    idx = _np.arange(len(uniq)); rng.shuffle(idx)
    picked = [uniq[i] for i in idx[:min(n_intents,len(uniq))]]
    parts = []
    for lab in picked:
        block = df[df[label_col].astype(str) == str(lab)]
        take = min(len(block), n_per_intent)
        if take == 0: continue
        parts.append(block.sample(take, random_state=random_state)[shown])
    if not parts:
        return df.sample(min(len(df), n_intents*n_per_intent), random_state=random_state)[shown].reset_index(drop=True)
    return pd.concat(parts, axis=0).reset_index(drop=True)

def show_split(name, df, **kw):
    try:
        from IPython.display import display
        print(f"\n=== {name} ===")
        display(sample_by_intent(df, **kw))
    except Exception as e:
        print(f"{name} sample error:", e)
