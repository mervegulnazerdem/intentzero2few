from __future__ import annotations
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_confusion_heatmap(cm_norm, labels, out_path, title: str | None = None, annot: bool = True):
    """
    Save a confusion matrix heatmap.
    - cm_norm: 2D list or np.ndarray (row-normalized values [0..1])
    - labels: list of class names (axis tick labels)
    - out_path: file path to save (directories will be created)
    """
    cm = np.asarray(cm_norm, dtype=float)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(labels)+2), max(5, 0.5*len(labels)+2)))
    sns.heatmap(cm, ax=ax, vmin=0.0, vmax=1.0, cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot=annot, fmt=".2f", cbar=True, square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
