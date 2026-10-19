"""Compact plots produced directly from recomputed scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import AnalysisConfig
from .data import TactileDataset
from .scores import METHOD_ORDER


COLORS = (
    "#2F6970",
    "#8EAEB0",
    "#C6AD82",
    "#8D9799",
    "#B07C91",
    "#748F83",
    "#C38A68",
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#899092",
            "axes.labelcolor": "#273033",
            "xtick.color": "#273033",
            "ytick.color": "#273033",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_roc_plot(
    dataset: TactileDataset,
    scores: dict[str, np.ndarray],
    config: AnalysisConfig,
    path: str | Path,
) -> None:
    configure_style()
    test = dataset.indices(split="test")
    labels = dataset.event_mask[:, config.start_frame + 1 :][test].reshape(-1)
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    ax.plot([0, 1], [0, 1], color="#B9BFBD", linestyle="--", linewidth=0.8)
    for method, color in zip(METHOD_ORDER, COLORS):
        values = scores[method][test, config.start_frame:].reshape(-1)
        fpr, tpr, _ = roc_curve(labels, values)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.0 if method == "Predictive mismatch" else 1.0,
            label=method,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend(loc="lower right", fontsize=6.2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_condition_auc_plot(table: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    pivot = table.pivot(index="condition", columns="method", values="test_AUC")
    order = ["weak_jam", "strong_jam", "unilateral_slip", "bilateral_slip"]
    pivot = pivot.loc[order, list(METHOD_ORDER)]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    image = ax.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="GnBu", aspect="auto")
    ax.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)), [x.replace("_", " ").title() for x in order])
    for row in range(len(order)):
        for column in range(len(METHOD_ORDER)):
            value = pivot.iloc[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, label="Held-out AUC", fraction=0.025, pad=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_representative_mismatch_plot(
    dataset: TactileDataset,
    predictive_score: np.ndarray,
    threshold: float,
    config: AnalysisConfig,
    path: str | Path,
) -> None:
    configure_style()
    candidates = dataset.indices(condition="weak_jam", split="test")
    index = int(candidates[0])
    score_frames = np.arange(1, dataset.n_frames)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(score_frames, predictive_score[index], color="#2F6970", linewidth=1.2)
    ax.axhline(threshold, color="#B8757A", linestyle="--", linewidth=1.0, label="Validation threshold")
    onset = int(dataset.event_onset_frame[index])
    end = int(dataset.event_end_frame[index])
    ax.axvspan(onset, end, color="#ECDCDD", alpha=0.65, label="Annotated event")
    ax.axvspan(0, config.start_frame, color="#F0F1EE", alpha=0.8, label="Initialization")
    ax.set(xlabel="Frame", ylabel="Predictive mismatch", xlim=(0, dataset.n_frames - 1))
    ax.set_title(f"Representative held-out trial {dataset.trial_id[index]}", loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

