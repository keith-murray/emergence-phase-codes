"""Functions to initialize models from `ctrnn-jax`."""

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.activation import tanh, sigmoid

from ctrnn_jax.model import CTRNNCell


def initialize_ctrnn(
    hidden_features=100, output_features=2, alpha=0.1, noise_const=0.05
):
    """Simple initialization function."""
    return nn.RNN(
        CTRNNCell(
            hidden_features=hidden_features,
            output_features=output_features,
            alpha=jnp.float32(alpha),
            noise_const=jnp.float32(noise_const),
        ),
        split_rngs={"params": False, "noise_stream": True},
    )


def initialize_ctrnn_with_activation(
    hidden_features=100,
    output_features=2,
    alpha=0.1,
    noise_const=0.05,
    activation_fn="tanh",
):
    """Initialization function with activation function."""
    activation_dict = {
        "tanh": tanh,
        "sigmoid": sigmoid,
    }
    return nn.RNN(
        CTRNNCell(
            hidden_features=hidden_features,
            output_features=output_features,
            alpha=jnp.float32(alpha),
            noise_const=jnp.float32(noise_const),
            activation_fn=activation_dict[activation_fn],
        ),
        split_rngs={"params": False, "noise_stream": True},
    )
