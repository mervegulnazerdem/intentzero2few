# intentzero2few

Hierarchical Zero→Few intent classification:
- Super-intent discovery (8–12 clusters)
- Zero-shot super-intent with OOS threshold
- Few-shot sub-intents
- Threshold calibration (macro-F1 sweep)
- Robust eval on clean vs polluted sets

## Open in Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/mervegulnazerdem/intentzero2few/blob/main/notebooks/Dissertation_merve_erdem_20092025.ipynb
)

## Outputs
One run → two roots with the same RUN_ID:
- `runs/<RUN_ID>/`  (raw): analytics/, logs/, figures/, artifacts/
- `reports/<RUN_ID>/` (curated): tables & figures for the thesis

Convenience symlinks:
- `runs/latest` and `reports/latest` → most recent run
