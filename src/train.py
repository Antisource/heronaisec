"""
Training utilities.

Responsibilities
----------------
1. Configure Hugging Face Trainer.
2. Fine-tune a model.
3. Save checkpoints.

Training is intentionally independent of:
- attack generation
- evaluation
- experiment orchestration
"""

from __future__ import annotations

from pathlib import Path
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    DataCollatorWithPadding,
)

# ============================================================================
# Trainer Construction
# ============================================================================

def build_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    tokenizer,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    ) -> Trainer:
    """
    Construct a Hugging Face Trainer.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=epochs,

        learning_rate=float(learning_rate),

        per_device_train_batch_size=batch_size,

        weight_decay=float(weight_decay),

        logging_strategy="epoch",

        save_strategy="epoch",

        report_to="none",

        seed=seed,
        
        #remove_unused_columns=False,
    )
    
    data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,      
        data_collator=data_collator,
    )


# ============================================================================
# Training
# ============================================================================

def train_model(
    trainer: Trainer,
    save_directory: str | Path,
     tokenizer,
):
    """
    Train a model and save the final checkpoint.
    """
    
    trainer.train()

    trainer.save_model(save_directory)
    
    tokenizer.save_pretrained(save_directory)

    return trainer.model