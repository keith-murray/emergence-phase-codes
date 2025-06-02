"""Pipeline to run DSA on trained models."""

import os

import numpy as np
import jax.numpy as jnp
from jax import random

import pandas as pd
from tqdm import tqdm
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

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
val_metrics = val_metrics.sort_values(by="job_id")

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
NUM_TRIALS = 500
JOBS_PATH = "./jobs"


def apply_pca_to_rates(rates_pre_pc, n_components=10):
    """Function to reduce dimensionality of rates."""
    n_trials, time_steps, n_neurons = rates_pre_pc.shape
    flattened = rates_pre_pc.reshape(
        -1, n_neurons
    )  # shape (n_trials * time_steps, n_neurons)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(
        flattened
    )  # shape (n_trials * time_steps, n_components)

    return reduced.reshape(n_trials, time_steps, n_components)


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
task.build_sequence_balanced_dataset(30, 15)
test_inputs, test_targets = task.generate_jax_tensor()

# Run all models on the test data
rates_list = []
alphas = []
markers = []
for _, row in tqdm(val_metrics.iterrows(), total=len(val_metrics)):
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
    if row["job_id"] < 100:
        params.deserialize(os.path.join(row["task_dir"], "model.bin"))
    else:
        params.deserialize(row["task_dir"])

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

    print(f"Testing accuracy: {accuracy}")

    # Apply PCA and store rates
    reduced_rates = apply_pca_to_rates(np.asarray(rates), n_components=10)
    rates_list.append(reduced_rates)
    alphas.append(row["alpha"])
    markers.append(row["activation_fn"])

# Step 4: Run DSA
print("Running DSA...")
dsa = DSA(
    rates_list,
    n_delays=10,
    delay_interval=5,
    rank=25,
    iters=1000,
    lr=1e-2,
    device="cpu",
    verbose=True,
)
similarity_matrix = dsa.fit_score()

# Save similarity matrix
np.save("./data/similarity_matrix.npy", similarity_matrix)

# Step 5: Visualize with MDS and similarity matrix
embedding = MDS(
    dissimilarity="precomputed",
    random_state=42,
).fit_transform(similarity_matrix)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot similarity matrix
im = ax1.imshow(similarity_matrix, cmap="viridis")
ax1.set_title("Similarity Matrix")
ax1.set_xlabel("Model Index")
ax1.set_ylabel("Model Index")
fig.colorbar(im, ax=ax1)

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
    color = cmap(norm(alphas[i]))
    marker = marker_dict.get(markers[i], "o")
    EDGE_COLOR = "black"

    # Special styling for job_id == 100
    if val_metrics.iloc[i]["job_id"] == 100:
        ax2.scatter(
            embedding[i, 0],
            embedding[i, 1],
            c="gold",
            marker="*",
            s=200,
            edgecolors=EDGE_COLOR,
            linewidths=1.5,
            zorder=5,
            label="Phase Code" if "Phase Code" not in seen else None,
        )
        seen.add("Phase Code")
    else:
        if label not in seen:
            ax2.scatter(
                embedding[i, 0],
                embedding[i, 1],
                c=[color],
                marker=marker,
                edgecolors=EDGE_COLOR,
                label=label,
            )
            seen.add(label)
        else:
            ax2.scatter(
                embedding[i, 0],
                embedding[i, 1],
                c=[color],
                marker=marker,
                edgecolors=EDGE_COLOR,
            )

    ax2.text(
        embedding[i, 0] + 0.01,
        embedding[i, 1] + 0.01,
        str(int(val_metrics.iloc[i]["job_id"])),
        fontsize=6,
        alpha=0.75,
    )

ax2.set_title("DSA MDS Embedding of CT-RNN Solutions")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")
ax2.grid(True)
ax2.legend(title="Activation Function")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2)
cbar.set_label("Alpha")

plt.tight_layout()
plt.savefig("./results/dsa_mds_embedding.png")
plt.show()
