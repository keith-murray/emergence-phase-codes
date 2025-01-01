# pylint: disable=too-many-locals
"""Tests for the Modulo 3 Arithmetic task."""

import jax.numpy as jnp
from jax import random

from recurrent_networks_oscillate.task import Modulo3Arithmetic


def test_create_trial():
    """
    Test the create_trial method of the Modulo3Arithmetic class.
    """
    key = random.PRNGKey(0)
    neurons = 10
    time = 50
    task = Modulo3Arithmetic(key, neurons=neurons, time=time)

    # Test case 1: Valid sequence with unique integers = 2
    integer_sequence = "221"
    unique_integers = 2
    input_array, output_array = task.create_trial(integer_sequence, unique_integers)

    # Validate input_array shape
    assert input_array.shape == (
        time,
        neurons,
    ), f"Unexpected input_array shape: {input_array.shape}"

    # Validate that random positions are placed correctly
    ball_indices = jnp.nonzero(input_array.sum(axis=1))[0]  # Find non-zero indices
    assert (
        len(ball_indices) == 6
    ), "Not enough ball positions were set (should be 6 due to duplicates)."

    # Validate that output_array is correctly labeled as -1
    assert output_array.shape == (
        1,
    ), f"Unexpected output_array shape: {output_array.shape}"
    assert (
        output_array.item() == -1.0
    ), f"Unexpected label for unique_integers=2: {output_array.item()}"

    # Test case 2: Valid sequence with unique integers = 3
    integer_sequence = "012"
    unique_integers = 3
    input_array, output_array = task.create_trial(integer_sequence, unique_integers)

    # Validate input_array shape
    assert input_array.shape == (
        time,
        neurons,
    ), f"Unexpected input_array shape: {input_array.shape}"

    # Validate that random positions are placed correctly
    ball_indices = jnp.nonzero(input_array.sum(axis=1))[0]
    assert (
        len(ball_indices) == 6
    ), "Not enough ball positions were set (should be 6 due to duplicates)."

    # Validate that output_array is correctly labeled as 1
    assert output_array.shape == (
        1,
    ), f"Unexpected output_array shape: {output_array.shape}"
    assert (
        output_array.item() == 1.0
    ), f"Unexpected label for unique_integers=3: {output_array.item()}"

    # Test case 3: Valid sequence with unique integers = 1
    integer_sequence = "111"
    unique_integers = 1
    input_array, output_array = task.create_trial(integer_sequence, unique_integers)

    # Validate input_array shape
    assert input_array.shape == (
        time,
        neurons,
    ), f"Unexpected input_array shape: {input_array.shape}"

    # Validate that random positions are placed correctly
    ball_indices = jnp.nonzero(input_array.sum(axis=1))[0]
    assert (
        len(ball_indices) == 6
    ), "Not enough ball positions were set (should be 6 due to duplicates)."

    # Validate that output_array is correctly labeled as 1
    assert output_array.shape == (
        1,
    ), f"Unexpected output_array shape: {output_array.shape}"
    assert (
        output_array.item() == 1.0
    ), f"Unexpected label for unique_integers=3: {output_array.item()}"


def test_generate_jax_tensor():
    """
    Test the generate_jax_tensor method of the Modulo3Arithmetic class.
    """
    key = random.PRNGKey(0)
    neurons = 10
    time = 50
    task = Modulo3Arithmetic(key, neurons=neurons, time=time)

    # Test Case 1: Validate Tensor Shapes
    data_dict_1 = {
        "012": [
            (jnp.zeros((time, neurons)), jnp.array([1.0])),
            (jnp.zeros((time, neurons)), jnp.array([1.0])),
        ],
        "120": [
            (jnp.zeros((time, neurons)), jnp.array([-1.0])),
            (jnp.zeros((time, neurons)), jnp.array([-1.0])),
        ],
        "201": [
            (jnp.zeros((time, neurons)), jnp.array([1.0])),
            (jnp.zeros((time, neurons)), jnp.array([1.0])),
        ],
    }
    features_tensor, labels_tensor = task.generate_jax_tensor(data_dict_1)
    assert features_tensor.shape == (
        6,
        time,
        neurons,
    ), f"Unexpected features_tensor shape: {features_tensor.shape}"
    assert labels_tensor.shape == (
        6,
        5,
        1,
    ), f"Unexpected labels_tensor shape: {labels_tensor.shape}"

    # Test Case 2: Validate Label Expansion
    data_dict_2 = {"012": [(jnp.zeros((time, neurons)), jnp.array([1.0]))]}
    _, labels_tensor = task.generate_jax_tensor(data_dict_2)
    expected_label = jnp.array([[1.0]] * 5)
    assert jnp.array_equal(
        labels_tensor[0], expected_label
    ), f"Unexpected label expansion: {labels_tensor[0]}"

    # Test Case 3: Validate Tensor Content Consistency
    mock_input = jnp.ones((time, neurons))
    mock_output = jnp.array([-1.0])
    data_dict_3 = {"012": [(mock_input, mock_output)]}
    features_tensor, labels_tensor = task.generate_jax_tensor(data_dict_3)
    assert jnp.array_equal(
        features_tensor[0], mock_input
    ), "Features tensor content mismatch."
    expected_label = jnp.tile(mock_output, (5, 1))
    assert jnp.array_equal(
        labels_tensor[0], expected_label
    ), "Labels tensor content mismatch."

    # Test Case 4: Validate Large Dataset
    data_dict_5 = {
        f"{i:03}": [
            (jnp.ones((time, neurons)) * i, jnp.array([1.0])) for _ in range(100)
        ]
        for i in range(10)
    }
    features_tensor, labels_tensor = task.generate_jax_tensor(data_dict_5)
    expected_trials = sum(len(trials) for trials in data_dict_5.values())
    assert features_tensor.shape == (
        expected_trials,
        time,
        neurons,
    ), f"Unexpected features_tensor shape: {features_tensor.shape}"
    assert labels_tensor.shape == (
        expected_trials,
        5,
        1,
    ), f"Unexpected labels_tensor shape: {labels_tensor.shape}"
