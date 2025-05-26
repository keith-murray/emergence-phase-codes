"""Pipeline to load json and train model."""

import os
import sys
import json
import csv

import jax.numpy as jnp
from jax import random

from ctrnn_jax.training import ModelParameters, create_train_state

from emergence_phase_codes.model import initialize_ctrnn_with_activation
from emergence_phase_codes.task import ModuloNArithmetic
from emergence_phase_codes.training import train_model_with_validation


# pylint: disable=too-many-locals
def main(params_path):
    """Train model given parameters in json config."""
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
        num_trials=2500,
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

    # Generate validation dataset
    task.num_trials = 500
    task.build_balanced_dataset()
    validation_dataset = task.generate_tf_dataset(16)

    # Train model with validation
    train_state, best_params, best_metrics, _ = train_model_with_validation(
        key,
        config["epochs"],
        train_state,
        dataset,
        validation_dataset,
        config["time_index"],
        config["rate_penalty"],
    )

    # Save best model
    model_params = ModelParameters(best_params)
    model_path = os.path.join(config["task_dir"], "model.bin")
    model_params.serialize(model_path)

    # Save best validation metrics to CSV
    metrics_path = os.path.join("./data/", "validation_metrics.csv")
    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                [
                    "seed",
                    "best_validation_loss",
                    "best_validation_accuracy",
                    "alpha",
                    "noise",
                    "activation_fn",
                    "model_path",
                ]
            )
        writer.writerow(
            [
                config["seed"],
                float(best_metrics["loss"]),
                float(best_metrics["accuracy"]),
                config["alpha"],
                config["noise"],
                config["activation_fn"],
                model_path,
            ]
        )


if __name__ == "__main__":
    params_file = sys.argv[1]
    main(params_file)
