# pylint: disable=not-callable,too-many-statements,duplicate-code
"""Pipeline to load model and generate figures."""

import os
import sys
import json

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import jax.numpy as jnp
from jax import random

from ctrnn_jax.training import ModelParameters, create_train_state
from ctrnn_jax.pca import compute_pca

from emergence_phase_codes.model import initialize_ctrnn_with_activation
from emergence_phase_codes.task import ModuloNArithmetic
from emergence_phase_codes.pca import add_data

from emergence_phase_codes.animations.output import OutputAnimator
from emergence_phase_codes.animations.pca import (
    PCATrajectoryAnimator,
    PCAPopulationAnimator,
)
from emergence_phase_codes.animations.utils import interpolate_colors


# pylint: disable=too-many-locals
def main(params_path):
    """Load model from a given path and generate figures."""
    with open(params_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    key = random.PRNGKey(config["seed"])

    # Setup task
    key, task_key = random.split(key)
    task = ModuloNArithmetic(
        task_key,
        3,
        congruent_number=0,
        time_length=50,
        num_trials=500,
        pulse_config={
            "num_pulses": 3,
            "pulse_window": 40,
            "pulse_buffer": 5,
            "pulse_gap": 5,
            "pulse_amplitude": 5,
        },
    )
    dataset = task.generate_tf_dataset(16)

    # Initialize model
    model = initialize_ctrnn_with_activation(
        hidden_features=100,
        output_features=1,
        alpha=config["alpha"],
        noise_const=config["noise"],
        activation_fn=config["activation_fn"],
    )

    # Create training state
    key, state_key = random.split(key)
    init_input = jnp.ones([16, 50, 3])
    train_state = create_train_state(
        state_key, model, config["learning_rate"], init_input
    )

    # Load parameters
    params = ModelParameters(train_state)
    params.deserialize(os.path.join(config["task_dir"], "model.bin"))

    # Compute PCA
    key, pca_key = random.split(key, num=2)
    model_behavior, pca = compute_pca(
        pca_key,
        train_state,
        params.params,
        dataset,
        2,
    )

    # Compute null rates
    key, test_key = random.split(key, num=2)
    _, rates_null = train_state.apply_fn(
        params.params, jnp.zeros((1, 50, 3)), rngs={"noise_stream": test_key}
    )
    rates_pc_null = pca.transform(rates_null[0, :, :])

    # Make congruent example input
    congruent_example_input, _ = task.create_trial_with_indices(
        jnp.array(
            [
                2,
                0,
                1,
            ]
        ),
        jnp.array([10, 20, 30]),
    )
    congruent_example_input = congruent_example_input[None, :, :]

    # Input congruent example into CT-RNN
    key, congruent_key = random.split(key, num=2)
    output_congruent, rates_congruent = train_state.apply_fn(
        params.params, congruent_example_input, rngs={"noise_stream": congruent_key}
    )

    # Add input, rates, and output to model_behavior
    model_behavior = add_data(
        model_behavior,
        pca,
        congruent_example_input,
        rates_congruent,
        output_congruent,
    )

    # Make incongruent example input
    incongruent_example_input, _ = task.create_trial_with_indices(
        jnp.array(
            [
                0,
                0,
                2,
            ]
        ),
        jnp.array([15, 25, 35]),
    )
    incongruent_example_input = incongruent_example_input[None, :, :]

    # Input congruent example into CT-RNN
    key, incongruent_key = random.split(key, num=2)
    output_incongruent, rates_incongruent = train_state.apply_fn(
        params.params, incongruent_example_input, rngs={"noise_stream": incongruent_key}
    )

    # Add input, rates, and output to model_behavior
    model_behavior = add_data(
        model_behavior,
        pca,
        incongruent_example_input,
        rates_incongruent,
        output_incongruent,
    )

    #
    output_congruent_dict = {-2: output_congruent[0, :, 0]}
    output_incongruent_dict = {-1: output_incongruent[0, :, 0]}

    blue_gradient = interpolate_colors("#7f7f7f", "#1f77b4", n_steps=50)
    orange_gradient = interpolate_colors("#7f7f7f", "#ff7f0e", n_steps=50)

    trajectory_indices = [-2, -1]
    trajectory_colors = {
        -2: blue_gradient,
        -1: orange_gradient,
    }
    classification_colors = {
        (1,): "tab:blue",
        (-1,): "tab:orange",
    }

    # Define colors for integer stimuli
    stimulus_colors = {0: "tab:green", 1: "tab:red", 2: "tab:purple"}

    # Decode integer sequences for example trials
    decoded_congruent = task.decode_integer_inputs(
        congruent_example_input[0, :, :],
    )
    decoded_incongruent = task.decode_integer_inputs(
        incongruent_example_input[0, :, :],
    )

    # Create the Figure
    fig_anim = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3)

    # Panel 1
    ax_1 = fig_anim.add_subplot(gs[0, 0])
    animator_1 = OutputAnimator(
        ax_1,
        output_congruent_dict,
        trajectory_colors,
        title="Example accept trial",
        stimulus_colors=stimulus_colors,
    )
    animator_1.color_integer_bars(
        decoded_congruent,
    )
    animator_1.add_stimulus_legend()
    ax_1.set_xticks([])
    ax_1.set_xlabel("")
    animator_1.figure()

    # Panel 2
    ax_2 = fig_anim.add_subplot(
        gs[
            1,
            0,
        ]
    )
    animator_2 = OutputAnimator(
        ax_2,
        output_incongruent_dict,
        trajectory_colors,
        title="Example reject trial",
        stimulus_colors=stimulus_colors,
    )
    animator_2.color_integer_bars(
        decoded_incongruent,
    )
    animator_2.figure()

    # Panel 3
    ax_3 = fig_anim.add_subplot(gs[:, 1])
    animator_3 = PCATrajectoryAnimator(
        ax_3,
        model_behavior,
        1,
        2,
        trajectory_indices,
        trajectory_colors,
        "Example PCA trajectories",
        stimulus_colors=stimulus_colors,
    )
    animator_3.color_integer_points(
        -2,
        decoded_congruent,
    )
    animator_3.color_integer_points(
        -1,
        decoded_incongruent,
    )
    animator_3.figure()

    # Panel 4
    ax_4 = fig_anim.add_subplot(
        gs[
            :,
            2,
        ]
    )
    animator_4 = PCAPopulationAnimator(
        ax_4,
        model_behavior,
        1,
        2,
        classification_colors,
        null_trajectory=rates_pc_null,
        highlight_indices=trajectory_indices,
        title="Trajectory endpoints in PCA space",
    )
    animator_4.figure()

    plt.tight_layout()
    plt.savefig(os.path.join(config["task_dir"], "model_dynamics.png"))
    plt.close()


if __name__ == "__main__":
    params_file = sys.argv[1]
    main(params_file)
