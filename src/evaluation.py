"""
Evaluation utilities for measuring backdoor persistence.

Responsibilities
----------------
1. Run inference.
2. Compute evaluation metrics.
3. Return structured results.

This module intentionally contains no training,
attack generation, or visualization logic.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase
 


# ============================================================================
# Inference
# ============================================================================

@torch.inference_mode()
def predict(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataset.

    Returns
    -------
    predictions:
        Predicted class labels.

    probabilities:
        Softmax probabilities.
    """

    device = next(model.parameters()).device
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    predictions = []
    probabilities = []

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        probs = torch.softmax(outputs.logits, dim=-1)

        predictions.extend(
            probs.argmax(dim=-1).cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

    return (
        np.asarray(predictions),
        np.asarray(probabilities),
    )


# ============================================================================
# Metrics
# ============================================================================

def compute_clean_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Classification accuracy on clean examples.
    """

    return accuracy_score(labels, predictions)


def compute_attack_success_rate(
    predictions: np.ndarray,
    target_label: int,
) -> float:
    """
    Fraction of triggered examples classified
    as the attacker's target label.
    """

    return float(
        np.mean(predictions == target_label)
    )


def compute_trigger_confidence(
    probabilities: np.ndarray,
    target_label: int,
) -> float:
    """
    Mean probability assigned to the target class
    on triggered inputs.
    """

    return float(
        probabilities[:, target_label].mean()
    )


def compute_retention_ratio(
    attack_success_rate: float,
    baseline_asr: float,
) -> float:
    """
    Normalized backdoor persistence.

    Retention Ratio = ASR(t) / ASR(0)
    """

    if baseline_asr == 0:
        return 0.0

    return attack_success_rate / baseline_asr


# ============================================================================
# Public Evaluation API
# ============================================================================

def evaluate_model(
    model: PreTrainedModel,
    clean_dataset: Dataset,
    triggered_dataset: Dataset,
    tokenizer,
    target_label: int,
    baseline_asr: float,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate a model on clean and triggered datasets.
    """

    clean_predictions, _ = predict(
        model=model,
        dataset=clean_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    triggered_predictions, triggered_probabilities = predict(
        model=model,
        dataset=triggered_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    clean_accuracy = compute_clean_accuracy(
        clean_predictions,
        np.asarray(clean_dataset["label"]),
    )

    attack_success_rate = compute_attack_success_rate(
        triggered_predictions,
        target_label,
    )

    trigger_confidence = compute_trigger_confidence(
        triggered_probabilities,
        target_label,
    )

    retention_ratio = compute_retention_ratio(
        attack_success_rate,
        baseline_asr,
    )

    return {
        "clean_accuracy": clean_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_trigger_confidence": trigger_confidence,
        "retention_ratio": retention_ratio,
    }