"""Create json config files for jobs."""

import os
import json
from itertools import product
from jax import random


# pylint: disable=too-many-locals
def main():
    """Function to generate job config files."""
    base_dir = "./jobs"
    os.makedirs(base_dir, exist_ok=True)

    # Candidate values for different hyperparameters
    alpha_values = [0.1, 0.5, 1.0]
    noise_values = [0.00, 0.05, 0.10]
    activation_functions = ["tanh", "relu"]
    job_combinations = list(product(alpha_values, noise_values, activation_functions))

    key = random.PRNGKey(0)

    job_id = 0
    for alpha, noise, activation_fn in job_combinations:
        for _ in range(10):  # 10 seeds per condition
            key, subkey = random.split(key)
            seed = int(random.randint(subkey, (), 0, 2**31 - 1))

            task_dir = os.path.join(base_dir, f"task_{job_id:04d}")
            os.makedirs(task_dir, exist_ok=True)

            params = {
                "job_id": job_id,
                "seed": seed,
                "alpha": alpha,
                "noise": noise,
                "activation_fn": activation_fn,
                "epochs": 1000,
                "learning_rate": 1e-4,
                "rate_penalty": 1e-4,
                "time_index": -3,
                "stop_acc": 0.99,
                "task_dir": task_dir,
            }

            with open(
                os.path.join(task_dir, "params.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(params, f, indent=2)

            job_id += 1


if __name__ == "__main__":
    main()
