"""Create json config files for jobs."""

import os
import json
import random


def main():
    """Function to generate job config files."""
    base_dir = "./jobs"
    os.makedirs(base_dir, exist_ok=True)

    # Candidate values for different hyperparameters
    alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    noise_values = [0.01, 0.05, 0.1]
    lr_values = [1e-3, 5e-4, 1e-4]
    rate_penalty_values = [1e-3, 1e-4, 1e-5]

    total_configs = 25
    random.seed(0)  # For reproducibility

    for task_id in range(total_configs):
        task_dir = os.path.join(base_dir, f"task_{task_id:04d}")
        os.makedirs(task_dir, exist_ok=True)

        # Randomly sample parameters
        alpha = random.choice(alpha_values)
        noise = random.choice(noise_values)
        learning_rate = random.choice(lr_values)
        rate_penalty = random.choice(rate_penalty_values)
        seed = task_id

        params = {
            "seed": seed,
            "alpha": alpha,
            "noise": noise,
            "epochs": 500,
            "learning_rate": learning_rate,
            "rate_penalty": rate_penalty,
            "time_index": -3,
            "task_dir": task_dir,
        }

        # pylint: disable=unspecified-encoding
        with open(os.path.join(task_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
