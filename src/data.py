"""
Dataset loading and preprocessing.

This module is intentionally model-agnostic and attack-agnostic.

Responsibilities
----------------
1. Load datasets.
2. Perform dataset-level preprocessing.
3. Tokenize text.
4. Prepare datasets for PyTorch training.

This module should never contain:
- Backdoor attacks
- Training logic
- Evaluation logic
"""

from __future__ import annotations

from typing import Iterable

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


# ============================================================================
# Expected dataset schema
# ============================================================================

# Keeping the schema in one place makes it easy to support new datasets later.
REQUIRED_COLUMNS = {"sentence", "label"}


# ============================================================================
# Dataset Loading
# ============================================================================

def load_text_dataset(
    dataset_name: str,
    subset: str,
) -> DatasetDict:
    """
    Load a text classification dataset from Hugging Face.

    Parameters
    ----------
    dataset_name:
        Dataset repository name (e.g. "glue").

    subset:
        Dataset configuration (e.g. "sst2").

    Returns
    -------
    DatasetDict
        Dataset containing train / validation / test splits.
    """

    return load_dataset(dataset_name, subset)


# ============================================================================
# Dataset Validation
# ============================================================================

def validate_dataset(dataset: DatasetDict) -> None:
    """
    Validate that every dataset split contains the required columns.

    Raises
    ------
    ValueError
        If a required column is missing.
    """

    for split_name, split in dataset.items():

        missing = REQUIRED_COLUMNS.difference(split.column_names)

        if missing:
            raise ValueError(
                f"{split_name} split is missing columns: {missing}"
            )


# ============================================================================
# Generic Preprocessing
# ============================================================================

def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    """
    Apply generic dataset preprocessing.

    These operations improve dataset quality without changing the
    scientific experiment.
    """

    def clean(example):

        # Remove leading/trailing whitespace.
        example["sentence"] = example["sentence"].strip()

        return example

    return dataset.map(clean)


# ============================================================================
# Tokenizer
# ============================================================================

def build_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the tokenizer associated with a pretrained model.
    """

    return AutoTokenizer.from_pretrained(model_name)


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_dataset(
    dataset,
    tokenizer,
):
    """
    Tokenize a dataset split.

    This function accepts either a Dataset or DatasetDict.
    """

    def tokenize(batch):

        return tokenizer(
            batch["sentence"],
            truncation=True,
        )

    return dataset.map(
        tokenize,
        batched=True,
    )


# ============================================================================
# PyTorch Formatting
# ============================================================================

def prepare_for_training(
    dataset,
):
    """
    Convert a tokenized dataset into PyTorch format.
    """

    return dataset.with_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label",
        ],
    )


# ============================================================================
# Public Pipeline
# ============================================================================

def prepare_dataset(
    dataset_name: str,
    subset: str,
    model_name: str,
):
    """
    Load and preprocess the dataset.

    Tokenization intentionally happens later,
    after attacks have been applied.
    """

    dataset = load_text_dataset(
        dataset_name,
        subset,
    )

    validate_dataset(dataset)

    dataset = preprocess_dataset(dataset)

    tokenizer = build_tokenizer(model_name)

    return (
        dataset["train"],
        dataset["validation"],
        tokenizer,
    )
 
 

