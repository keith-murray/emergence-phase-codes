"""Launch jobs"""

import os
import subprocess
from multiprocessing import Pool

N_PARALLEL = 6
BASE_DIR = "./jobs"


def train_job(task_dir):
    """Function to call `pipeline.py` with pointer to json config."""
    params_path = os.path.join(BASE_DIR, task_dir, "params.json")
    subprocess.run(["python", "scripts/training_pipeline.py", params_path], check=False)


def main():
    """Run all jobs in `jobs/` directory."""
    task_dirs = sorted(
        [
            d
            for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("task_")
        ]
    )
    with Pool(N_PARALLEL) as pool:
        pool.map(train_job, task_dirs)


if __name__ == "__main__":
    main()
