"""Pipeline to run DSA on trained models."""

import os
import json

import numpy as np
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import MDS

from ctrnn_jax.training import create_train_state, ModelParameters
from DSA import DSA

from emergence_phase_codes.model import initialize_ctrnn
from emergence_phase_codes.task import ModuloNArithmetic
from emergence_phase_codes.training import compute_classification_accuracy


# Load validation metrics
val_metrics = pd.read_csv("./data/validation_metrics.csv")

# Parameters
MOD = 3
HIDDEN_SIZE = 100
OUTPUT_SIZE = 1
PULSE_CONFIG = {
    "num_pulses": 3,
    "pulse_window": 40,
    "pulse_buffer": 5,
    "pulse_gap": 5,
    "pulse_amplitude": 5,
}
TIME_LENGTH = 50
NUM_TRIALS = 100
JOBS_PATH = "./jobs"

# Create a shared M3A test dataset
key = random.PRNGKey(42)
key, subkey = random.split(key)
task = ModuloNArithmetic(
    subkey,
    MOD,
    congruent_number=0,
    time_length=TIME_LENGTH,
    num_trials=NUM_TRIALS,
    pulse_config=PULSE_CONFIG,
)
test_inputs, test_targets = task.generate_jax_tensor()

# Create the base model
model = initialize_ctrnn(
    hidden_features=HIDDEN_SIZE,
    output_features=OUTPUT_SIZE,
    alpha=1.0,
    noise_const=0,  # For analyzing dynamics, noise is always set to 0
)


# Run all models on the test data
rates_list = []
for job_dir in tqdm(sorted(os.listdir(JOBS_PATH))):
    # Load seed from params.json in same folder
    params_path = os.path.join(JOBS_PATH, job_dir, "params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    seed = config["seed"]
    alpha = config["alpha"]

    # Retrieve validation accuracy
    val_row = val_metrics[val_metrics["seed"] == seed]
    if val_row.empty:
        continue
    csv_accuracy = val_row["best_validation_accuracy"].values[0]

    # Re-initialize model
    model = initialize_ctrnn(
        hidden_features=HIDDEN_SIZE,
        output_features=OUTPUT_SIZE,
        alpha=alpha,
        noise_const=0,  # For analyzing dynamics, noise is always set to 0
    )
    key, train_state_key = random.split(key, num=2)
    model_state = create_train_state(
        train_state_key,
        model,
        1e-4,
        jnp.ones([16, TIME_LENGTH, MOD]),
    )

    # Load parameters
    params = ModelParameters(model_state)
    params.deserialize(os.path.join(JOBS_PATH, job_dir, "model.bin"))

    # Run model
    key, subkey = random.split(key)
    # pylint: disable=not-callable
    output, rates = model_state.apply_fn(
        params.params, test_inputs, rngs={"noise_stream": subkey}
    )

    accuracy = compute_classification_accuracy(
        output[:, -1, -1], test_targets[:, -1, -1]
    )

    if accuracy > 0.95 and csv_accuracy > 0.95:
        rates_list.append(np.asarray(rates))
    else:
        continue

# Step 4: Run DSA
print("Running DSA...")
dsa = DSA(rates_list, verbose=True)
similarity_matrix = dsa.fit_score()

# Step 5: Visualize with MDS
embedding = MDS(dissimilarity="precomputed").fit_transform(1 - similarity_matrix)

plt.figure(figsize=(6, 6))
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("DSA MDS Embedding of CT-RNN Solutions")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/dsa_mds_embedding.png")
plt.show()
