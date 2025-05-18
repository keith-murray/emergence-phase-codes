"""Helper functions for PCA calculations."""

import jax.numpy as jnp


def add_data(model_behavior, pca, model_input, model_rates, model_output):
    """
    Applies PCA to model_rates and appends data to model_behavior.

    Args:
        model_behavior (dict): Dictionary from `compute_pca` function.
        pca (sklearn.decomposition.PCA): Fitted PCA object.
        model_input (jnp.ndarray): Task inputs (shape: [n_trials, n_time, input_dim]).
        model_rates (jnp.ndarray): CT-RNN firing rates (shape: [n_trials, n_time, n_units]).
        model_output (jnp.ndarray): Model outputs (shape: [n_trials, n_time, output_dim]).

    Returns:
        dict: Updated `model_behavior` dictionary containing:
            - "inputs" (jnp.ndarray): Task inputs.
            - "rates" (jnp.ndarray): CT-RNN firing rates.
            - "rates_pc" (jnp.ndarray): Principal components of `rates`.
            - "outputs" (jnp.ndarray): Model outputs.
    """
    # Apply PCA transformation to model_rates
    model_rates_pc = pca.transform(model_rates.reshape(-1, model_rates.shape[-1]))
    model_rates_pc = model_rates_pc.reshape(
        model_rates.shape[0], model_rates.shape[1], -1
    )

    # Append data to model_behavior
    model_behavior["inputs"] = jnp.concatenate(
        [model_behavior["inputs"], model_input], axis=0
    )
    model_behavior["rates"] = jnp.concatenate(
        [model_behavior["rates"], model_rates], axis=0
    )
    model_behavior["rates_pc"] = jnp.concatenate(
        [model_behavior["rates_pc"], model_rates_pc], axis=0
    )
    model_behavior["outputs"] = jnp.concatenate(
        [model_behavior["outputs"], model_output], axis=0
    )

    return model_behavior
