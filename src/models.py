"""
Model construction and checkpoint management.

Responsibilities
----------------
1. Build pretrained sequence classification models.
2. Save trained checkpoints.
3. Reload trained checkpoints.

This module intentionally contains no training,
evaluation, or attack logic.
"""

from __future__ import annotations
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
)


# ============================================================================
# Model Construction
# ============================================================================

def build_model(
    model_name: str,
    num_labels: int,
) ->PreTrainedModel:
    """
    Construct a pretrained sequence classification model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.

    num_labels:
        Number of output classes.

    Returns
    -------
    PreTrainedModel
        Initialized model ready for fine-tuning.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_model(
    model: PreTrainedModel,
    output_dir: str | Path,
) -> None:
    """
    Save a trained model checkpoint.
    """

    model.save_pretrained(output_dir)
    
# Save tokenizer separately in data.py / train.py, not here.
# save_pretrained() already saves the model configuration.


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_model(
    checkpoint_dir: str | Path,
) -> PreTrainedModel:
    """
    Load a previously saved model checkpoint.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
    )