"""
Visualization utilities.

Responsibilities
----------------
1. Create publication-quality figures.
2. Save figures to disk.

This module intentionally contains no:
- training
- evaluation
- attack logic
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# Global Plot Style
# ============================================================================

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "legend.frameon": False,
    }
)


# ============================================================================
# Backdoor Persistence Plot
# ============================================================================

def plot_backdoor_persistence(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot backdoor persistence across intervention strengths.
    """
    required = {
    "clean_ratio",
    "attack_success_rate",
    "clean_accuracy",
    "mean_trigger_confidence",
    }

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )
        
    results = results.sort_values(
    "clean_ratio"
    ).reset_index(drop=True)

    fig, ax1 = plt.subplots(constrained_layout=True)

    # Primary metrics
    ax1.plot(
        results["clean_ratio"],
        results["attack_success_rate"],
        marker="o",
        linewidth=2,
        label="Attack Success Rate",
    )

    ax1.plot(
        results["clean_ratio"],
        results["mean_trigger_confidence"],
        marker="s",
        linewidth=2,
        label="Trigger Confidence",
    )

    ax1.set_xlabel("Clean Fine-Tuning Ratio")
    ax1.set_ylabel("Metric Value")
    ax1.set_ylim(0.0, 1.05)

    # Secondary axis for clean accuracy
    ax2 = ax1.twinx()

    ax2.plot(
        results["clean_ratio"],
        results["clean_accuracy"],
        linestyle="--",
        linewidth=2,
        label="Clean Accuracy",
    )

    ax2.set_ylabel("Clean Accuracy")

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc="best")

    plt.title("Backdoor Persistence Under Progressive Clean Fine-Tuning")


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")

    plt.close(fig)


# ============================================================================
# Reviewer Intervention Plot
# ============================================================================

def plot_reviewer_intervention(
    results: pd.DataFrame,
    reviewer_ratio: float,
    output_path: str | Path,
) -> None:
    """
    Highlight the reviewer intervention on the persistence curve.
    """
    
    required = {
    "clean_ratio",
    "attack_success_rate",
    "clean_accuracy",
    "mean_trigger_confidence",
    }

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )
        
    results = results.sort_values(
    "clean_ratio"
    ).reset_index(drop=True)
    
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(
        results["clean_ratio"],
        results["attack_success_rate"],
        marker="o",
        linewidth=2,
        label="Baseline",
    )

    reviewer = results[
        results["clean_ratio"] == reviewer_ratio
    ]

    ax.scatter(
        reviewer["clean_ratio"],
        reviewer["attack_success_rate"],
        s=100,
        marker="*",
        label="Reviewer Intervention",
    )

    ax.set_xlabel("Clean Fine-Tuning Ratio")
    ax.set_ylabel("Attack Success Rate")

    ax.legend()

    plt.title("Effect of Reviewer Intervention")


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")

    plt.close(fig)