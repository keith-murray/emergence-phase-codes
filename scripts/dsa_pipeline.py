"""Pipeline to run DSA on trained models."""

import os

import numpy as np
import jax.numpy as jnp
from jax import random

import pandas as pd
from tqdm import tqdm
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

from ctrnn_jax.training import create_train_state, ModelParameters
from DSA import DSA

from emergence_phase_codes.model import initialize_ctrnn_with_activation
from emergence_phase_codes.task import ModuloNArithmetic
from emergence_phase_codes.training import compute_classification_accuracy


# Load validation metrics
val_metrics = pd.read_csv("./data/validation_metrics.csv")
val_metrics = val_metrics.sort_values(by="seed")

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
NUM_TRIALS = 1000
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

# Run all models on the test data
rates_list = []
alphas = []
markers = []
for _, row in tqdm(val_metrics.iterrows(), total=len(val_metrics)):
    # Retrieve validation accuracy
    csv_accuracy = row["best_validation_accuracy"]

    # Re-initialize model
    model = initialize_ctrnn_with_activation(
        hidden_features=HIDDEN_SIZE,
        output_features=OUTPUT_SIZE,
        alpha=row["alpha"],
        noise_const=0,  # For analyzing dynamics, noise is always set to 0
        activation_fn=row["activation_fn"],
    )
    key, train_state_key = random.split(key, num=2)
    model_state = create_train_state(
        train_state_key,
        model,
        1e-4,  # Just an example learning rate
        jnp.ones([16, TIME_LENGTH, MOD]),
    )

    # Load parameters
    params = ModelParameters(model_state)
    params.deserialize(os.path.join(row["task_dir"], "model.bin"))

    # Run model
    key, subkey = random.split(key)
    # pylint: disable=not-callable
    output, rates = model_state.apply_fn(
        params.params, test_inputs, rngs={"noise_stream": subkey}
    )

    # Compute test accuracy
    accuracy = compute_classification_accuracy(
        output[:, -1, -1], test_targets[:, -1, -1]
    )

    print(f"Validation accuracy: {csv_accuracy}\nTesting accuracy: {accuracy}")
    rates_list.append(np.asarray(rates))
    alphas.append(row["alpha"])
    markers.append(row["activation_fn"])

# Step 4: Run DSA
print("Running DSA...")
dsa = DSA(rates_list, rank=30, verbose=True, iters=1000, lr=1e-2)
similarity_matrix = dsa.fit_score()

# Step 5: Visualize with MDS
embedding = MDS(dissimilarity="precomputed").fit_transform(1 - similarity_matrix)

fig, ax = plt.subplots(figsize=(6, 6))

marker_dict = {
    "tanh": "o",
    "relu": "s",
}
# pylint: disable=no-member
cmap = cm.cool
norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))
seen = set()
for i in range(len(embedding)):
    label = markers[i]
    if label not in seen:
        ax.scatter(
            embedding[i, 0],
            embedding[i, 1],
            c=[cmap(norm(alphas[i]))],
            marker=marker_dict.get(markers[i], "o"),
            label=label,
            edgecolors="black",
        )
        seen.add(label)
    else:
        ax.scatter(
            embedding[i, 0],
            embedding[i, 1],
            c=[cmap(norm(alphas[i]))],
            marker=marker_dict.get(markers[i], "o"),
            edgecolors="black",
        )

plt.title("DSA MDS Embedding of CT-RNN Solutions")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Activation Function")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Alpha")
plt.savefig("./results/dsa_mds_embedding.png")
plt.show()
