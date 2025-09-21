# Run Report

- RUN_ID: 20250921-123702

## Flow (ASCII)

```text
FLOW (Zero→Few Robust Pipeline)
┌─────────┐
│  Input  │  CLINC_OOS (train/val/test + OOS)
└───┬─────┘
    │
    ├─▶ Split & Export (data/clinc150.json)
    │
    ├─▶ EDA (counts, wordcloud)  → runs/analytics, runs/figures
    │
    ├─▶ Prep:
    │     ├─ Polluted {val,test} (mix OOS)
    │     ├─ Augmented train (classic) → ≥30K
    │     └─ Noisy {train,test} (emoji/slang/typo)
    │
    ├─▶ Discovery (TF-IDF desc → ST embeddings → K-means) → super-intents
    │
    ├─▶ Zero-shot (super centroids) + τ calibration (val_polluted)
    │
    ├─▶ Evaluate: clean / noisy / polluted
    │
    ├─▶ Baselines (TFIDF+LR, BERT-Linear) + τ
    │
    ├─▶ Few-shot K={1,5,10}
    │
    └─▶ Reports (tables + figs + artifacts) → reports/<RUN_ID>/
```

## Flow (Mermaid)

```mermaid
flowchart TD
A[CLINC_OOS raw] --> B[Export CLINC JSON]
B --> C[EDA: stats + wordcloud]
B --> D[Prep: polluted val/test]
B --> E[Prep: augmented train >=30K]
B --> F[Prep: noisy train/test]
E --> G[Discovery: super-intents (K-means)]
G --> H[Zero-shot centroids]
D --> I[tau calibration on val_polluted]
H --> J[Evaluate: clean/noisy/polluted]
F --> J
B --> K[Baselines TFIDF+LR, BERT-Linear + tau]
B --> L[Few-shot K=1/5/10]
J --> M[Reports: zs_summary, heatmaps]
K --> M
L --> M
```

## Key Artifacts (present)
- intent_descriptions.csv
- zs_summary.csv
- baseline_robustness.csv
- fewshot_summary.csv
- benchmark_summary.csv
- summary_metrics.csv
- run_meta.json
- zs_confmat_clean.png
- zs_confmat_noisy.png
- zs_confmat_polluted.png

## Key Artifacts (missing / to be generated)
- wordcloud_train.png
