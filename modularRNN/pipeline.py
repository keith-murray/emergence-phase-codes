import jax.numpy as jnp
from jax import random
from flax import linen as nn
from modularRNN.model import CTRNNCell
from modularRNN.task import ModularArithmeticTask
from modularRNN.training import create_train_state, train_model
from modularRNN.analysis import compute_cumulative_variance

from functools import partial

import json
import os
import sys

task_id = sys.argv[1] # Here is the $LLSUB_RANK slurm argument
experiment_folder = sys.argv[2] # Here is a non-slurm argument, this folder is the same across the entire job
task_folder = os.path.join(experiment_folder, f"task_{task_id}")

json_path = os.path.join(task_folder, "params.json")
with open(json_path, 'r') as f:
    json_params = json.load(f)

seed = json_params.get('seed', 0)
alpha = json_params.get('alpha', 0.1)
noise = json_params.get('noise', 0.05)
pulse_mean = json_params.get('pulse_mean', 12)
mod_set = json_params.get('mod_set', [5,])
training_trials = json_params.get('training_trials', 3200)
testing_trials = json_params.get('testing_trials', 640)
train_batch_size = json_params.get('train_batch_size', 128)
lr = json_params.get('lr', 0.001)
epochs = json_params.get('epochs', 500)
weight_decay = json_params.get('weight_decay', 0.0001)
l2_penalty = json_params.get('l2_penalty', 0.0001)
trial_length = json_params.get('trial_length', 100)

key = random.PRNGKey(seed)

mod_set = jnp.array(mod_set)
pulse_distribution = partial(random.poisson, lam=pulse_mean)

key, subkey = random.split(key)
modtask = ModularArithmeticTask(
    subkey, 
    training_trials, 
    testing_trials, 
    train_batch_size, 
    mod_set, 
    pulse_distribution, 
    trial_length,
)
training_dataset, testing_dataset = modtask.tf_datasets()

key, subkey = random.split(key)
modtask_long = ModularArithmeticTask(
    subkey, 
    640, 
    640, 
    640, 
    mod_set, 
    partial(random.poisson, lam=2*pulse_mean), 
    500,
)
_, testing_dataset_long = modtask_long.tf_datasets()

features = 100
alpha = jnp.float32(alpha)
noise = jnp.float32(noise)

ctrnn = nn.RNN(CTRNNCell(features=features, alpha=alpha, noise=noise, out_shape=10,))

key, subkey = random.split(key)
state = create_train_state(ctrnn, subkey, lr, weight_decay, trial_length)

key, subkey = random.split(key)
results = train_model(
    subkey, 
    state, 
    training_dataset, 
    testing_dataset, 
    testing_dataset_long,
    epochs,
    l2_penalty,
)

results["final_params"].serialize(os.path.join(task_folder, 'final_params.bin'))
results["min_test_loss_params"].serialize(os.path.join(task_folder, 'test_params.bin'))
results["min_long_loss_params"].serialize(os.path.join(task_folder, 'long_params.bin'))
results["metrics_history"].save_to_csv(os.path.join(task_folder, 'metrics_history.csv'))

key, subkey = random.split(key)
compute_cumulative_variance(
    ctrnn, 
    results["min_test_loss_params"].params, 
    training_dataset, 
    subkey, 
    os.path.join(task_folder, 'cumulative_variance.npy'),
)