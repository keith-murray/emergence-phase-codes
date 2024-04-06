import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct
from flax import serialization
import optax
from clu import metrics
import pandas as pd

@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss') # type: ignore

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, subkey, learning_rate, weight_decay, trial_length):
    """Creates an initial `TrainState`."""
    params = module.init(subkey, jnp.ones([1, trial_length, 20]))['params'] # (batch, time, inputs)
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )

@jax.jit
def train_step(state, batch, subkey, l2_penalty):
    """Train for a single step."""
    def loss_fn(params):
        output, rates = state.apply_fn({'params': params}, batch[0], init_key=subkey)
        loss_task = optax.squared_error(output, batch[1]).mean()
        loss_rates = jnp.float32(l2_penalty) * optax.squared_error(rates).mean()
        return loss_task + loss_rates
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def compute_metrics(state, batch, subkey, l2_penalty):
    output, rates = state.apply_fn({'params': state.params}, batch[0], init_key=subkey)
    loss_task = optax.squared_error(output, batch[1]).mean()
    loss_rates = jnp.float32(l2_penalty) * optax.squared_error(rates).mean()
    loss = loss_task + loss_rates
    metric_updates = state.metrics.single_from_model_output(loss=loss,)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

class MetricsHistory:
    def __init__(self, metrics):
        self.history = {metric: [] for metric in metrics}
        
    def append(self, metric, value):
        self.history[metric].append(value)

    def latest(self, metric):
        return self.history[metric][-1]
        
    def save_to_csv(self, save_loc):
        df = pd.DataFrame(self.history)
        df.to_csv(save_loc, index=False)

    def load_from_csv(self, save_loc):
        loaded_df = pd.read_csv(save_loc)
        self.history = loaded_df.to_dict(orient='list')
    
    def print_latest_metrics(self):
        for metric, values in self.history.items():
            latest_value = values[-1] if values else "N/A"
            print(f"{metric}: {latest_value}")
        print("\n")

class ModelParameters:
    def __init__(self, state):
        self.params = {'params': state.params}
    
    def serialize(self, save_loc):
        bytes_output = serialization.to_bytes(self.params)
        with open(save_loc, 'wb') as f:
            f.write(bytes_output)
    
    def deserialize(self, save_loc):
        with open(save_loc, 'rb') as f:
            bytes_output = f.read()
        # Need to already have example `params` loaded to saved params
        self.params = serialization.from_bytes(self.params, bytes_output)

def compute_metrics_and_update_history(subkey, state, batch, metric_prefix, metrics_history, l2_penalty):
    """
    Compute metrics for a given dataset batch and update the metrics history.

    Parameters:
        state: The model state.
        batch: The dataset batch.
        subkey: The random subkey.
        metric_prefix: A string prefix to use for the metrics (e.g., 'train', 'test').
        metrics_history: An instance of MetricsHistory.
        l2_penalty (float)
    """
    new_state = compute_metrics(state, batch, subkey, l2_penalty)
    for metric, value in new_state.metrics.compute().items():
        metrics_history.append(f'{metric_prefix}_{metric}', value)

def train_model(key, state, train_ds, test_ds, epochs, l2_penalty):
    """
    Trains a model using provided datasets and records various performance metrics.

    Parameters:
        key (random.PRNGKey): The JAX random key for stochastic operations.
        state (object): The initial state of the model, including parameters.
        train_ds (Dataset): The dataset for training the model.
        test_ds (Dataset): The dataset for testing the model.
        epochs (int): The number of training epochs.
        l2_penalty (float)

    Returns:
        tuple: 
    """
    metrics_history = MetricsHistory([
        'train_loss',
        'test_loss',
    ])

    min_test_loss = float('inf')
    min_test_loss_params = None
    
    test_batch = list(test_ds.as_numpy_iterator())[0]

    for epoch in range(epochs):

        for _, batch in enumerate(train_ds.as_numpy_iterator()):
            key, subkey = random.split(key)
            state = train_step(state, batch, subkey, l2_penalty)
            state = compute_metrics(state, batch, subkey, l2_penalty)

        for metric, value in state.metrics.compute().items():
            metrics_history.append(f'train_{metric}', value)
        state = state.replace(metrics=state.metrics.empty())

        key, subkey = random.split(key)
        compute_metrics_and_update_history(subkey, state, test_batch, 'test', metrics_history, l2_penalty)

        # Check and store parameters for minimum test_loss
        current_test_loss = metrics_history.latest('test_loss')
        if current_test_loss < min_test_loss:
            min_test_loss = current_test_loss
            min_test_loss_params = ModelParameters(state)

        if (epoch+1) % 50 == 0:
            print(f'Metrics after epoch {epoch+1}:')
            metrics_history.print_latest_metrics()
    
    model_params = ModelParameters(state)
    
    return {
        "final_params": model_params,
        "min_test_loss_params": min_test_loss_params,
        "metrics_history": metrics_history,
    }