from functools import partial
from typing import (
  Any,
  Callable,
  Optional,
)

import jax.numpy as jnp
from jax import random
from flax.linen import initializers, RNNCellBase
from flax.linen.activation import relu
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact
from flax.typing import (
  Dtype,
  Initializer,
)

class CTRNNCell(RNNCellBase):
    """
    A continuous-time recurrent neural network (CTRNN) cell
    that is discritized using a first-order Euler scheme.

    Parameters:
        features (int): The number of output features.
        alpha (jnp.float32): The ratio of dt to tau.
        noise (jnp.float32): The noise multiplier.
        out_shape (int): The number of output dimensions.
    """
    features: int
    alpha: jnp.float32
    noise: jnp.float32
    out_shape: int
    activation_fn: Callable[..., Any] = relu
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs,):
        """
        Compute the next state and output of the cell given the current state and input.

        Parameters:
            carry (tuple): A tuple containing the current hidden state (jnp.ndarray) and a JAX random key.
            inputs (jnp.ndarray): The input tensor for the current time step.

        Returns:
            tuple: A tuple containing two elements:
                - A new carry tuple containing the updated hidden state (jnp.ndarray) and a new JAX random key.
                - A tuple containing the output tensor (jnp.ndarray) and the rate tensor (jnp.ndarray).
        """
        h, key = carry
        hidden_features = h.shape[-1]

        dense_h = partial(
            Dense,
            features=hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
        )
        dense_i = partial(
            Dense,
            features=hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        dense_o = partial(
            Dense,
            features=self.out_shape,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )

        noise_shape = h.shape
        key, subkey = random.split(key)

        leak = (jnp.float32(1.0) - self.alpha) * h
        recurrent = dense_h(name='h')(h)
        stimulus = dense_i(name='i')(inputs)
        noise = jnp.sqrt(jnp.float32(2.0)/self.alpha) * self.noise * random.normal(subkey, noise_shape)

        new_h = leak + self.alpha * self.activation_fn(recurrent + stimulus + noise)

        z = dense_o(name='o')(new_h)
        
        return (new_h, key), (z, new_h)

    def initialize_carry(self, key, input_shape,):
        """
        Initialize the state (carry) of the CTRNN cell.

        Parameters:
            key (random.PRNGKey): A JAX random key for initialization.
            input_shape (tuple[int]): The shape of an input sample, not including the time dimension.

        Returns:
            tuple: A carry tuple containing the initial hidden state (jnp.ndarray) and the JAX random key.
        """
        batch_dims = input_shape[:1]
        mem_shape = batch_dims + (self.features,)
        key, subkey = random.split(key)
        h = self.carry_init(subkey, mem_shape, self.param_dtype,)
        
        return h, key

    @property
    def num_feature_axes(self,):
        return 1