"""
Attack implementations.

Each attack transforms a clean dataset into an attacked dataset.

This module is intentionally independent of:
- model architectures
- training pipelines
- evaluation

Its only responsibility is dataset transformation.
"""

from __future__ import annotations

from datasets import Dataset


# ============================================================================
# Triggering Injection
# ============================================================================


def inject_trigger(
    text: str,
    trigger: str,
) -> str:
    """
    Insert a trigger token at the beginning of a text sequence.

    Parameters
    ----------
    text:
        Original input sentence.

    trigger:
        Trigger token used for poisoning.

    Returns
    -------
    str
        Triggered sentence.
    """

    return f"{trigger} {text}"


# ============================================================================
# Poisoning a single example
# ============================================================================


def poison_example(
    example: dict,
    trigger: str,
    target_label: int,
) -> dict:
    """
    Convert one clean example into a poisoned example.
    """

    poisoned = example.copy()

    poisoned["sentence"] = inject_trigger(
        poisoned["sentence"],
        trigger,
    )

    poisoned["label"] = target_label

    return poisoned


# ============================================================================
# Poisoning a dataset
# ============================================================================


def poison_dataset(
    dataset: Dataset,
    poison_rate: float,
    trigger: str,
    target_label: int,
    seed: int,
) -> Dataset:
    """
    Poison a fraction of the training dataset.

    Only the selected examples are modified.
    """

    shuffled = dataset.shuffle(seed=seed)

    n_poison = int(len(shuffled) * poison_rate)

    def poison(
        example,
        index,
    ):

        if index >= n_poison:
            return example

        return poison_example(
            example,
            trigger,
            target_label,
        )

    return shuffled.map(
        poison,
        with_indices=True,
    )
    
    
# ============================================================================
# A Public interface
# ============================================================================


def apply_attack(
    dataset: Dataset,
    attack_type: str,
    **kwargs,
) -> Dataset:
    """
    Apply an attack to a dataset.

    This dispatcher allows future attack implementations to
    share a common interface.
    """

    if attack_type.lower() == "badnet":

        return poison_dataset(
            dataset=dataset,
            poison_rate=kwargs["poison_rate"],
            trigger=kwargs["trigger"],
            target_label=kwargs["target_label"],
            seed=kwargs["seed"],
        )

    raise ValueError(
        f"Unsupported attack: {attack_type}"
    )