"""Training functions for CT-RNNs on the Modulo3Arithmetic task."""

import jax
import jax.numpy as jnp
import optax

from ctrnn_jax.training import Metrics, TrainState


def create_train_state(key, module, learning_rate, norm_clip, init_array):
    """Creates an initial `TrainState`."""
    params = module.init(key, init_array)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(norm_clip),
        optax.adamw(
            learning_rate,
        ),
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
    )


@jax.jit
def train_step(
    key,
    state,
    batch,
):
    """Train for a single step."""

    def loss_fn(params):
        output, rates = state.apply_fn(
            {"params": params}, batch[0], rngs={"noise_stream": key}
        )
        loss_task = optax.squared_error(output[:, -5:, :], batch[1]).mean()
        loss_rates = (
            jnp.float32(0.0001) * optax.squared_error(rates).mean() * jnp.float32(0.01)
        )
        return loss_task + loss_rates

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_accuracy(output, label):
    """Compute accuracy for the Modulo3Arithmetic task."""
    condition_positive = output > jnp.float32(0.5)
    condition_negative = output < jnp.float32(-0.5)

    output_clipped = jnp.where(
        condition_positive,
        jnp.float32(1.0),
        jnp.where(condition_negative, jnp.float32(-1.0), jnp.float32(0.0)),
    )
    matched = output_clipped == label
    accuracy = jnp.mean(matched)

    return accuracy


@jax.jit
def compute_metrics(key, state, batch):
    """Compute metrics for the Modulo3Arithmetic task."""
    output, rates = state.apply_fn(
        {"params": state.params}, batch[0], rngs={"noise_stream": key}
    )

    loss_task = optax.squared_error(output[:, -5:, :], batch[1]).mean()
    loss_rates = (
        jnp.float32(0.0001) * optax.squared_error(rates).mean() * jnp.float32(0.01)
    )
    loss = loss_task + loss_rates

    accuracy = compute_accuracy(output[:, -1, -1], batch[1][:, -1, -1])

    metric_updates = state.metrics.single_from_model_output(
        loss=loss, accuracy=accuracy
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state
