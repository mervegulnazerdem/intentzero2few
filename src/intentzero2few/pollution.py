from __future__ import annotations
import re, random, numpy as pdnp, pandas as pd
np = pdnp  # alias to avoid accidental shadowing

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokenize_en(s: str):
    return [w.lower() for w in _WORD_RE.findall(str(s))]

def _jaccard(a: set, b: set):
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def _collect_vocab(df, text_col: str = "text", sample_n: int = 10000):
    vocab = set()
    for t in df[text_col].head(sample_n):
        vocab.update(_tokenize_en(t))
    return vocab

def generate_fallback_negatives_en(n: int = 100, seed: int = 42,
                                   avoid_like_df: pd.DataFrame | None = None,
                                   max_trials_per_item: int = 10,
                                   gibberish_ratio: float = 0.3) -> pd.DataFrame:
    rng = random.Random(seed)
    topics_en = [
        "weather forecast","stock market trends","movie showtimes","soccer results","traffic updates",
        "air quality index","music recommendations","travel itineraries","baking recipes","breaking news headlines",
        "scientific discoveries","astrology readings","wildlife conservation","space exploration","art exhibitions",
        "quantum computing","medieval history","mountaineering safety tips","aquarium maintenance","vintage car auctions"
    ]
    question_templates = [
        "What is the latest on {topic}?","Can you summarize {topic} for me?","Where can I find resources about {topic}?",
        "What are some key facts about {topic}?","Could you give me a brief overview of {topic}?"
    ]
    command_templates = [
        "List three insights about {topic}.","Provide a short guide on {topic}.","Generate bullet points covering {topic}.",
        "Outline the main considerations for {topic}.","Give me a quick tip related to {topic}."
    ]
    gibberish_templates = [
        "lorem ipsum placeholder text {i}","random token sequence {i}","unrelated noise string {i}",
        "/// dummy //// content //// {i}","??? gibberish line {i}"
    ]
    inscope_vocab = set()
    if (avoid_like_df is not None) and (len(avoid_like_df) > 0) and ("text" in avoid_like_df.columns):
        inscope_vocab = _collect_vocab(avoid_like_df, text_col="text")

    def make_sentence(i: int) -> str:
        if rng.random() < (1.0 - gibberish_ratio):
            topic = rng.choice(topics_en); tpl = rng.choice(question_templates + command_templates)
            return tpl.format(topic=topic)
        else:
            tpl = rng.choice(gibberish_templates); return tpl.format(i=i)

    texts = []
    for i in range(n):
        trials = 0
        while True:
            s = make_sentence(i)
            if not inscope_vocab:
                texts.append(s); break
            jac = _jaccard(set(_tokenize_en(s)), inscope_vocab)
            if jac < 0.3 or trials >= max_trials_per_item:
                texts.append(s); break
            trials += 1
    return pd.DataFrame({"text": texts, "intent": ["__NEG__"] * n})

def make_polluted_test(test_df: pd.DataFrame, oos_df: pd.DataFrame | None = None,
                       ratio: float = 0.3, seed: int = 42,
                       fallback_random_negatives: bool = False):
    rng = np.random.RandomState(seed)
    n_oos = int(len(test_df) * ratio)
    if (oos_df is not None) and (len(oos_df) > 0):
        oos_sample = oos_df.sample(n=min(n_oos, len(oos_df)), random_state=rng)
        pol = (pd.concat([test_df.assign(is_oos=0), oos_sample.assign(is_oos=1)], ignore_index=True)
                 .sample(frac=1, random_state=rng).reset_index(drop=True))
        return pol, oos_sample
    if fallback_random_negatives:
        neg = generate_fallback_negatives_en(n=n_oos, seed=seed, avoid_like_df=test_df)
        pol = (pd.concat([test_df.assign(is_oos=0), neg.assign(is_oos=1)], ignore_index=True)
                 .sample(frac=1, random_state=rng).reset_index(drop=True))
        return pol, neg
    return test_df.copy(), None

def make_polluted_test_debug(test_df: pd.DataFrame, oos_df: pd.DataFrame | None = None,
                             ratio: float = 0.3, seed: int = 42,
                             fallback_random_negatives: bool = False):
    rng = np.random.RandomState(seed)
    n_oos = int(len(test_df) * ratio)
    print("\n--- POLLUTION DEBUG ---")
    print("Original test_df size:", len(test_df))
    print("OOS df provided?", "YES" if (oos_df is not None and len(oos_df) > 0) else "NO")
    print("Fallback random negatives?", "YES" if fallback_random_negatives else "NO")
    print("Planned OOS additions:", n_oos)
    if (oos_df is not None) and (len(oos_df) > 0):
        oos_sample = oos_df.sample(n=min(n_oos, len(oos_df)), random_state=rng)
        print("OOS sample size:", len(oos_sample))
        pol = (pd.concat([test_df.assign(is_oos=0), oos_sample.assign(is_oos=1)], ignore_index=True)
                 .sample(frac=1, random_state=rng).reset_index(drop=True))
        print("Polluted size after merge:", len(pol))
        return pol, oos_sample
    if fallback_random_negatives:
        neg = pd.DataFrame({"text": [f"random unrelated text {i}" for i in range(n_oos)],
                            "intent": ["__NEG__"] * n_oos})
        print("Random negatives generated:", len(neg))
        pol = (pd.concat([test_df.assign(is_oos=0), neg.assign(is_oos=1)], ignore_index=True)
                 .sample(frac=1, random_state=rng).reset_index(drop=True))
        print("Polluted size after merge:", len(pol))
        return pol, neg
    print("No pollution applied. Returning original test_df.")
    return test_df.copy(), None
