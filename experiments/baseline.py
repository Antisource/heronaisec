"""
Baseline experiment.

Project:
Beyond Binary Evaluation of Backdoor Inheritance

Responsibilities
----------------
1. Load experiment configuration.
2. Prepare datasets.
3. Train the initial backdoored model.
4. Measure backdoor persistence under progressively
   stronger clean fine-tuning.

This file intentionally contains NO model implementation,
NO attack implementation and NO evaluation logic.

It simply orchestrates the experiment.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils import (
    load_config,
    seed_everything,
    ensure_dir,
    timestamp,
    create_run_directory,
   )

from src.data import (
    tokenize_dataset,
    prepare_for_training,
)

from src.data import prepare_dataset

from src.attacks import apply_attack

from src.models import build_model, load_model

from src.train import (
    build_trainer,
    train_model,
)

from src.interventions import apply_clean_intervention

from src.evaluation import evaluate_model

from src.visualization import (
    plot_backdoor_persistence,
    plot_reviewer_intervention,
)

LOAD_FROM_CHECKPOINT = True # Deafult Value is False as we require training
CHECKPOINT_RUN = "baseline\\20260719_124348"

# ============================================================================
# Experiment Setup
# ============================================================================

def setup_experiment(config_path: str):
    """
    Prepare everything required before training begins.

    Returns
    -------
    config
        Parsed YAML configuration.

    train_dataset
        Clean training split.

    test_dataset
        Clean evaluation split.

    tokenizer
        Tokenizer associated with the chosen model.
    """

    config = load_config(config_path)

    seed_everything(
        config["experiment"]["seed"]
    )

    ensure_dir(
        config["experiment"]["output_dir"]
    )

    train_dataset, test_dataset, tokenizer = prepare_dataset(
        dataset_name=config["dataset"]["name"],
        subset=config["dataset"]["subset"],
        model_name=config["model"]["name"],
    )

    return (
        config,
        train_dataset,
        test_dataset,
        tokenizer,
    )


# ============================================================================
# Initial Backdoor Training
# ============================================================================

def train_backdoored_model(
    config: dict,
    train_dataset,
    tokenizer,
    run_dir,
):
    """
    Train the initial backdoored classifier.

    Returns
    -------
    model
        Fine-tuned backdoored model.
    """
    # -------------------------------------------------------------------------
    # Prototype subset
    # Use a smaller training set for faster experimentation.
    # Remove this line for the final full-scale experiment.
    # -------------------------------------------------------------------------

    train_dataset = (
    train_dataset
    .shuffle(seed=config["experiment"]["seed"])
    .select(range(5000))
)

    poisoned_train = apply_attack(
        dataset=train_dataset,
        attack_type=config["attack"]["type"],
        poison_rate=config["attack"]["poison_rate"],
        trigger=config["attack"]["trigger"],
        target_label=config["attack"]["target_label"],
        seed=config["experiment"]["seed"],
    )
    
    poisoned_train = tokenize_dataset(
    poisoned_train,
    tokenizer,
    )

    poisoned_train = prepare_for_training(
    poisoned_train,
    )

    model = build_model(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )

    checkpoint_dir = checkpoint_root / "baseline"
    
    
    trainer = build_trainer(
        model=model,
        train_dataset=poisoned_train,
        tokenizer=tokenizer,
        output_dir=str(checkpoint_dir),
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        weight_decay=config["training"]["weight_decay"],
        seed=config["experiment"]["seed"],
    )

    model = train_model(
        trainer=trainer,
        save_directory=checkpoint_dir,
        tokenizer=tokenizer,
    )

    return model

# ============================================================================
# Progressive Intervention Experiment
# ============================================================================

def run_progressive_intervention(
    config: dict,
    baseline_model,
    train_dataset,
    test_dataset,
    tokenizer,
    run_dir,
    checkpoint_root,
) -> pd.DataFrame:
    """
    Measure how backdoor behaviour changes as progressively
    larger amounts of clean fine-tuning data are introduced.

    Returns
    -------
    pd.DataFrame
        Metrics collected at every intervention strength.
    """

    # ------------------------------------------------------------------------
    # Build a fully-triggered evaluation set.
    #
    # Every example is poisoned so that Attack Success Rate (ASR)
    # measures whether the backdoor is still active.
    # ------------------------------------------------------------------------
  
    clean_eval = tokenize_dataset(
    test_dataset,
    tokenizer,
)

    clean_eval = prepare_for_training(
        clean_eval,
    )

    triggered_eval = apply_attack(
        dataset=test_dataset,
        attack_type=config["attack"]["type"],
        poison_rate=1.0,          # Trigger every test example
        trigger=config["attack"]["trigger"],
        target_label=config["attack"]["target_label"],
        seed=config["experiment"]["seed"],
    )

    triggered_eval = tokenize_dataset(
        triggered_eval,
        tokenizer,
    )

    triggered_eval = prepare_for_training(
    triggered_eval,
)

    # ------------------------------------------------------------------------
    # Baseline evaluation.
    #
    # We first measure the backdoored model before any intervention.
    # This baseline ASR becomes the denominator for Retention Ratio.
    # ------------------------------------------------------------------------

    baseline_metrics = evaluate_model(
        model=baseline_model,
        clean_dataset=clean_eval,
        triggered_dataset=triggered_eval,
        tokenizer=tokenizer,
        target_label=config["attack"]["target_label"],
        baseline_asr=1.0,
        batch_size=config["training"]["batch_size"],
    )

    baseline_asr = baseline_metrics["attack_success_rate"]

    results = []

    # Store the baseline measurement.
    baseline_metrics["clean_ratio"] = 0.0
    baseline_metrics["experiment"] = "baseline"

    results.append(baseline_metrics)

    # ------------------------------------------------------------------------
    # Progressive clean fine-tuning.
    # ------------------------------------------------------------------------

    for ratio in config["intervention"]["clean_ratios"]:

        # Skip the baseline because it has already been evaluated.
        if ratio == 0.0:
            continue

        print(f"\nRunning intervention at {ratio:.0%} clean data...")

        checkpoint_dir = (
            checkpoint_root
            / f"checkpoint_{int(ratio * 100)}"
                )

        if LOAD_FROM_CHECKPOINT:

            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_dir}"
                )

            print(f"Loading {ratio:.0%} checkpoint...")

            intervention_model = load_model(checkpoint_dir)

        else:

            intervention_model = apply_clean_intervention(
                model=baseline_model,
                clean_dataset=train_dataset,
                ratio=ratio,
                tokenizer=tokenizer,
                output_dir=checkpoint_dir,
                epochs=config["intervention"]["epochs"],
                learning_rate=config["training"]["learning_rate"],
                batch_size=config["training"]["batch_size"],
                weight_decay=config["training"]["weight_decay"],
                seed=config["experiment"]["seed"],
            )

        metrics = evaluate_model(
            model=intervention_model,
            clean_dataset=clean_eval,
            triggered_dataset=triggered_eval,
            tokenizer=tokenizer,
            target_label=config["attack"]["target_label"],
            baseline_asr=baseline_asr,
            batch_size=config["training"]["batch_size"],
        )

        metrics["clean_ratio"] = ratio
        metrics["experiment"] = "clean_intervention"

        results.append(metrics)

    return pd.DataFrame(results)

# ============================================================================
# Result Persistence
# ============================================================================

def save_results(
    config: dict,
    results: pd.DataFrame,
    run_dir,
     
) -> None:
    """
    Persist experiment outputs.

    Outputs
    -------
    results.csv
        Tabular metrics for every intervention.

    persistence_curve.(png/pdf)
        Main figure used in the report.

    reviewer_intervention.(png/pdf)
        Placeholder until reviewer experiments
        are implemented.
    """
    # ------------------------------------------------------------------------
    # Save tabular results
    # ------------------------------------------------------------------------

    results.to_csv(
        run_dir / "metrics.csv",
        index=False,
    )
    
    # ------------------------------------------------------------------------
    # Create figures directory
    # ------------------------------------------------------------------------

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)


    # ------------------------------------------------------------------------
    # Generate publication-quality figures
    # ------------------------------------------------------------------------

    if config["visualization"]["save_figures"]:

        plot_backdoor_persistence(
            results=results,
            output_path=figure_dir / "persistence_curve",
        )

        if config["reviewer"]["enabled"]:
            plot_reviewer_intervention(
                results=results,
                reviewer_ratio=config["reviewer"]["checkpoint_ratio"],
                output_path=figure_dir / "reviewer_intervention",
            )


# ============================================================================
# Experiment Orchestration
# ============================================================================



def run_baseline(
    config_path: str = "configs/baseline.yaml",
) -> None:
    """
    Execute the complete baseline experiment.
    """

    print("=" * 72)
    print("Beyond Binary Evaluation of Backdoor Inheritance")
    print("=" * 72)

    (
        config,
        train_dataset,
        test_dataset,
        tokenizer,
    ) = setup_experiment(config_path)
    
    # Create experiment output directory
    run_dir = (
    Path(config["experiment"]["output_dir"])
    / f"{config['experiment']['name']}_{timestamp()}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_root = (
    Path(config["experiment"]["output_dir"])
    / CHECKPOINT_RUN
    / "checkpoints"
)

    print("\nTraining initial backdoored model...")
    
    if LOAD_FROM_CHECKPOINT:
        checkpoint_dir = checkpoint_root / "baseline_checkpoint"
         
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_dir}"
            )

        print("Loading baseline checkpoint...")
        baseline_model = load_model(checkpoint_dir)
        
    else:
        baseline_model = train_backdoored_model(
            config=config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            run_dir=run_dir,
        )

    print("\nRunning progressive intervention study...")

    
    results = run_progressive_intervention(
            config=config,
            baseline_model=baseline_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            run_dir=run_dir,
            checkpoint_root=checkpoint_root,
            )

    print("\nSaving outputs...")

    save_results(
        config=config,
        run_dir=run_dir,
        results=results,
    )

    print("\nExperiment completed successfully.")


