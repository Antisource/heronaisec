"""
Intervention methods for studying backdoor inheritance.

Responsibilities
----------------
1. Apply clean fine-tuning.
2. Apply reviewer-based intervention.
3. Return an updated model.

This module intentionally does NOT:
- compute metrics
- plot figures
- orchestrate experiments
"""

from __future__ import annotations

from copy import deepcopy

from datasets import Dataset
from transformers import PreTrainedModel

from src.train import build_trainer, train_model

from src.data import (
    tokenize_dataset,
    prepare_for_training,
)

# ============================================================================
# Clean Fine-tuning
# ============================================================================

def fine_tune_clean(
    model: PreTrainedModel,
    train_dataset: Dataset,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    tokenizer,
) -> PreTrainedModel:
    """
    Fine-tune a model on clean data.
    """

    trainer = build_trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
    )

    return train_model(
        trainer=trainer,
        save_directory=output_dir,
        tokenizer=tokenizer,
    )
    

# ============================================================================
# Progressive Clean Intervention
#

def apply_clean_intervention(
    model: PreTrainedModel,
    clean_dataset: Dataset,
    tokenizer,
    ratio: float,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> PreTrainedModel:
    """
    Apply progressive clean fine-tuning.

    The intervention strength is controlled by `ratio`.
    """
    if ratio == 0:
        return deepcopy(model)

    subset_size = int(len(clean_dataset) * ratio)

    clean_subset = (
    clean_dataset
    .shuffle(seed=seed)
    .select(range(subset_size))
    )
    
    clean_subset = tokenize_dataset(
    clean_subset,
    tokenizer,
    )

    clean_subset = prepare_for_training(
        clean_subset,
    )

    model_copy = deepcopy(model)

    return fine_tune_clean(
        model=model_copy,
        train_dataset=clean_subset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
    )
 
    
#
# Reviewer Intervention (Scaffold) To be done Later
#
'''
def apply_reviewer_intervention(
    model: PreTrainedModel,
    clean_dataset: Dataset,
    reviewer_model: PreTrainedModel,
):
    """
    Placeholder for reviewer-model intervention.

    This will be implemented after the evaluation
    pipeline is complete.
    """

    raise NotImplementedError(
        "Reviewer intervention will be implemented in Phase 2."
    )
'''