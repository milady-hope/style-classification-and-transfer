from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import LABEL_NAMES

LABELS = [LABEL_NAMES[0], LABEL_NAMES[1]]


def plot_cm(y_true, y_pred, title: str, filename: str = None):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=LABELS, yticklabels=LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def summary_table(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def plot_cv_folds(all_cv: dict, filename: str = None, n_folds: int = 5):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    n_models = len(all_cv)
    x = np.arange(n_folds)
    w = 0.8 / n_models

    for idx, (metric, title) in enumerate(
        [("f1", "F1 (macro) по фолдам GroupKFold"),
         ("acc", "Accuracy по фолдам GroupKFold")]
    ):
        ax = axes[idx]
        for i, (name, df_cv) in enumerate(all_cv.items()):
            vals = df_cv[metric].values
            mean_v = vals.mean()
            bars = ax.bar(x + i * w, vals, w, label=name, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
            ax.axhline(mean_v, ls="--", alpha=0.4,
                       color=bars[0].get_facecolor(), linewidth=1)
            ax.text(n_folds - 0.3, mean_v + 0.003, f"{mean_v:.4f}",
                    fontsize=8, color=bars[0].get_facecolor(), fontweight="bold")

        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
        ax.set_title(title)
        ax.set_ylabel(metric.upper() if metric == "f1" else "Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylim(0.70, 1.05)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
