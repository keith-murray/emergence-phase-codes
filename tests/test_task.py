"""Tests for the ModuloNArithmetic task."""

from itertools import product

import jax.numpy as jnp
from jax import random

from emergence_phase_codes.task import ModuloNArithmetic


def test_create_trial_shapes_and_labels():
    """Test that trial inputs and labels have correct shapes and values."""
    key = random.PRNGKey(0)
    mod = 3
    task = ModuloNArithmetic(key, mod)
    sequence = jnp.array([0, 1, 2])
    inputs, labels = task.create_trial(sequence)

    assert inputs.shape == (task.time_length, mod), "Input shape incorrect"
    assert labels.shape == (task.time_length, 1), "Label shape incorrect"
    expected_label = 1.0 if sequence.sum() % mod == task.congruent_number else -1.0
    assert labels[0] == expected_label, "Label value incorrect"


def test_create_trial_with_indices():
    """Test that pulses are correctly set at provided indices."""
    key = random.PRNGKey(1)
    mod = 4
    task = ModuloNArithmetic(key, mod)
    sequence = jnp.array([1, 2, 0])
    pulse_indices = jnp.array([5, 15, 30])

    inputs, _ = task.create_trial_with_indices(sequence, pulse_indices)
    for idx, val in zip(pulse_indices, sequence):
        assert (
            inputs[idx, val] == task.pulse_amplitude
        ), f"Pulse not correctly set at index {idx}"
        assert (
            inputs[idx + 1, val] == task.pulse_amplitude
        ), f"Pulse not correctly set at index {idx+1}"


def test_sampling_check():
    """Test that sampling check correctly identifies invalid and valid gaps."""
    key = random.PRNGKey(2)
    task = ModuloNArithmetic(key, mod=3, pulse_config={"pulse_gap": 5})
    assert task.sampling_check(jnp.array([0, 3, 10])) is True
    assert task.sampling_check(jnp.array([0, 6, 15])) is False


def test_build_balanced_dataset():
    """Test that the generated dataset is balanced across class labels."""
    key = random.PRNGKey(3)
    task = ModuloNArithmetic(key, mod=3, num_trials=100)
    pos = sum(1 for _, y in task.dataset if y[0] == 1.0)
    neg = sum(1 for _, y in task.dataset if y[0] == -1.0)
    assert pos == neg == 50, "Dataset not class balanced"


def test_build_sequence_balanced_dataset():
    """Test that the dataset contains the correct number of trials per sequence class."""
    key = random.PRNGKey(4)
    mod = 2
    task = ModuloNArithmetic(key, mod)
    task.build_sequence_balanced_dataset(trials_per_pos_seq=3, trials_per_neg_seq=5)
    pos_sequences = [
        s
        for s in product(range(mod), repeat=task.num_pulses)
        if sum(s) % mod == task.congruent_number
    ]
    neg_sequences = [
        s
        for s in product(range(mod), repeat=task.num_pulses)
        if sum(s) % mod != task.congruent_number
    ]
    assert len(task.dataset) == len(pos_sequences) * 3 + len(neg_sequences) * 5


def test_generate_jax_tensor_shapes():
    """Test that generated JAX tensors have the correct shape."""
    key = random.PRNGKey(5)
    task = ModuloNArithmetic(key, mod=3, num_trials=10)
    features, labels = task.generate_jax_tensor()
    assert features.shape == (10, task.time_length, 3), "Feature tensor shape incorrect"
    assert labels.shape == (10, task.time_length, 1), "Label tensor shape incorrect"


def test_decode_integer_inputs():
    """Test that integer decoding from input array works as expected."""
    key = random.PRNGKey(6)
    task = ModuloNArithmetic(key, mod=3)
    input_array = jnp.zeros((task.time_length, 3)).at[10, 2].set(1)
    decoded = task.decode_integer_inputs(input_array)
    assert decoded[10] == 2
    assert all(d is None for i, d in enumerate(decoded) if i != 10)
