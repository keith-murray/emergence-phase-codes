"""Training functions."""

from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
import optax

import matplotlib.pyplot as plt


@partial(jax.jit, static_argnames=["index"])
def train_step(
    key,
    state,
    batch,
    index,
    rate_penalty,
):
    """Train for a single step with modified time index and rate penalty."""

    def loss_fn(params):
        output, rates = state.apply_fn(
            {"params": params}, batch[0], rngs={"noise_stream": key}
        )
        error = optax.squared_error(output, batch[1])[:, index:, :]
        activity = optax.squared_error(rates)
        return error.mean() + rate_penalty * activity.mean()

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_classification_accuracy(
    output,
    label,
):
    """
    Accuracy function for classification tasks.

    Need to add test cases.
    """
    condition_positive = output > jnp.float32(0.5)
    condition_negative = output < jnp.float32(-0.5)

    output_intermediate_clip = jnp.where(
        condition_negative, jnp.float32(-1.0), jnp.float32(0.0)
    )
    output_clipped = jnp.where(
        condition_positive, jnp.float32(1.0), output_intermediate_clip
    )
    matched = output_clipped == label
    accuracy = jnp.mean(matched)

    return accuracy


@partial(jax.jit, static_argnames=["index"])
def compute_metrics(
    key,
    state,
    batch,
    index=0,
):
    """Compute metrics after training step."""
    output, _ = state.apply_fn(
        {"params": state.params}, batch[0], rngs={"noise_stream": key}
    )
    loss = optax.squared_error(output, batch[1])[:, index:, :].mean()
    accuracy = compute_classification_accuracy(output[:, -1, -1], batch[1][:, -1, -1])
    metric_updates = state.metrics.single_from_model_output(
        loss=loss, accuracy=accuracy
    )
    metrics_updated = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics_updated)
    return state


# pylint: disable=too-many-arguments,too-many-positional-arguments
def train_model(
    key,
    epochs,
    state,
    tf_dataset,
    time_index,
    rate_penalty,
    tqdm_disable=False,
):
    """Train model on task for a given number of epochs."""
    # Initalize metrics
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
    }

    # Train model
    for _ in tqdm(range(epochs), disable=tqdm_disable):

        # Perform backpropagation
        for _, batch in enumerate(tf_dataset.as_numpy_iterator()):
            key, train_key, metrics_key = random.split(key, num=3)
            state = train_step(
                train_key,
                state,
                batch,
                time_index,
                rate_penalty,
            )
            state = compute_metrics(
                metrics_key,
                state,
                batch,
                index=time_index,
            )

        # Compute and store metrics
        for metric, value in state.metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value.item())

        # Clear metrics
        state = state.replace(metrics=state.metrics.empty())

    return state, metrics_history


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def train_model_with_validation(
    key,
    epochs,
    state,
    tf_dataset_train,
    tf_dataset_val,
    time_index,
    rate_penalty,
    tqdm_disable=False,
    early_stop_accuracy=None,
):
    """Train model on task with validation set and track best model."""

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_val_metrics = {}

    for _ in tqdm(range(epochs), disable=tqdm_disable):
        # Training step
        for _, batch in enumerate(tf_dataset_train.as_numpy_iterator()):
            key, train_key, metrics_key = random.split(key, num=3)
            state = train_step(
                train_key,
                state,
                batch,
                time_index,
                rate_penalty,
            )
            state = compute_metrics(metrics_key, state, batch, index=time_index)

        # Compute and store training metrics
        train_metrics = state.metrics.compute()
        metrics_history["train_loss"].append(train_metrics["loss"].item())
        metrics_history["train_accuracy"].append(train_metrics["accuracy"].item())
        state = state.replace(metrics=state.metrics.empty())

        # Validation step
        for _, batch in enumerate(tf_dataset_val.as_numpy_iterator()):
            key, metrics_key = random.split(key)
            state = compute_metrics(metrics_key, state, batch, index=time_index)

        val_metrics = state.metrics.compute()
        val_loss = val_metrics["loss"].item()
        val_accuracy = val_metrics["accuracy"].item()
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_accuracy)
        state = state.replace(metrics=state.metrics.empty())

        # Track best parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = {"loss": val_loss, "accuracy": val_accuracy}
            best_state = state

        # Early Stopping logic
        if early_stop_accuracy is not None and val_accuracy >= early_stop_accuracy:
            print(
                f"Stopping early: val acc {val_accuracy:.4f} >= thresh {early_stop_accuracy}"
            )
            break

    return state, best_state, best_val_metrics, metrics_history


def plot_training_loss_accuracy(epochs, metrics_history, save_loc=False, show=True):
    """Standard plotting function for training loss and accuracy."""
    # Visualize training metrics
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Loss over training epochs
    axs[0].plot(
        jnp.arange(epochs),
        metrics_history["train_loss"],
        label="Train",
        color="tab:blue",
    )
    if "val_loss" in metrics_history:
        axs[0].plot(
            jnp.arange(epochs),
            metrics_history["val_loss"],
            label="Validation",
            color="tab:orange",
        )

    axs[0].set_title("Loss over training epochs")
    axs[0].set_xlabel("Training epoch")
    axs[0].set_ylabel("Mean squared error")
    axs[0].legend()

    # Panel 2: Accuracy over training epochs
    axs[1].plot(
        jnp.arange(epochs),
        metrics_history["train_accuracy"],
        label="Train",
        color="tab:blue",
    )
    if "val_accuracy" in metrics_history:
        axs[1].plot(
            jnp.arange(epochs),
            metrics_history["val_accuracy"],
            label="Validation",
            color="tab:orange",
        )

    axs[1].set_title("Accuracy over training epochs")
    axs[1].set_xlabel("Training epoch")
    axs[1].set_ylabel("Percentage correct")
    axs[1].legend()

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc)

    if show:
        plt.show()
    else:
        plt.close()
