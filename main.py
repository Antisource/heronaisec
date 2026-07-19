"""
Repository entry point.

Runs the baseline experiment defined in
experiments/baseline.py.
"""

from experiments.baseline import run_baseline
import torch

def main() -> None:
    """
    Execute the baseline experiment.
    """
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    run_baseline()


if __name__ == "__main__":
    main()