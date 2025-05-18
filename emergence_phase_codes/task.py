# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
"""Implementation of modular arithmetic task."""

from itertools import product

from jax import random
import jax.numpy as jnp
import tensorflow as tf


class ModuloNArithmetic:
    """
    A task class for generating trials of a Modulo-N arithmetic task.

    The task presents pulses of integers (in [0, N-1]) using one-hot encoding.
    The model must determine whether the sum of integers in a sequence (mod N)
    equals a specified `congruent_number`. Trials are balanced between positive
    (sum == congruent_number) and negative (sum != congruent_number) classes.

    The class supports flexible configuration of pulse timing, spacing, amplitude,
    and generates balanced datasets for supervised learning.
    """

    def __init__(
        self,
        key,
        mod,
        congruent_number=0,
        time_length=50,
        num_trials=1000,
        pulse_config=None,
    ):
        """
        Initialize the ModuloNArithmetic task.

        Args:
            key (PRNGKey): JAX random number generator key.
            mod (int): Modulo value; determines the number of unique integer inputs.
            congruent_number (int): The target value for modulo congruency (default: 0).
            time_length (int): Number of time steps per trial.
            num_trials (int): Total number of trials to generate (split evenly between classes).
            pulse_config (dict, optional): Dictionary with keys:
                - 'num_pulses' (int): Number of pulses per trial.
                - 'pulse_window' (int): Time window size for pulse placement.
                - 'pulse_buffer' (int): Buffer offset before pulses start.
                - 'pulse_gap' (int): Minimum gap between pulses.
                - 'pulse_amplitude' (float): Value of the one-hot pulse (default: 1).
        """
        self.key = key
        self.mod = mod
        self.congruent_number = congruent_number
        self.time_length = time_length
        self.num_trials = num_trials

        # Pulse config
        if pulse_config is None:
            self.num_pulses = 3
            self.pulse_window = time_length
            self.pulse_buffer = 0
            self.pulse_gap = 0
            self.pulse_amplitude = 1
        else:
            self.num_pulses = pulse_config.get("num_pulses", 3)
            self.pulse_window = pulse_config.get("pulse_window", time_length)
            self.pulse_buffer = pulse_config.get("pulse_buffer", 0)
            self.pulse_gap = pulse_config.get("pulse_gap", 0)
            self.pulse_amplitude = pulse_config.get("pulse_amplitude", 1)

        # Build dataset
        self.dataset = []
        self.build_balanced_dataset()

    def split_key(self):
        """
        Split the internal PRNGKey to produce a new subkey.

        Returns:
            PRNGKey: A new random subkey.
        """
        self.key, subkey = random.split(self.key)
        return subkey

    def sampling_check(self, pulse_indices):
        """
        Check whether the generated pulse indices meet the minimum gap requirement.

        Args:
            pulse_indices (jnp.ndarray): Array of pulse time indices.

        Returns:
            bool: True if pulse gap is violated, False otherwise.
        """
        if pulse_indices.shape[0] > 1:
            pulse_gaps = pulse_indices[1:] - pulse_indices[:-1]
            return (jnp.min(pulse_gaps) < self.pulse_gap).item()
        return False

    def create_trial(self, sequence):
        """
        Create a single trial with a specific sequence of integers.

        Args:
            sequence (jnp.ndarray): 1D array of length `num_pulses`, integers in [0, M-1].

        Returns:
            tuple:
                - input_array (jnp.ndarray): Shape (time_length, mod), one-hot pulses over time.
                - output_array (jnp.ndarray): Shape (time_length, 1), tiled label (+1 or -1).
        """
        invalid_indices = True
        while invalid_indices:
            subkey = self.split_key()
            pulse_indices = random.choice(
                subkey, self.pulse_window, shape=(self.num_pulses,), replace=False
            )
            pulse_indices = jnp.sort(pulse_indices) + self.pulse_buffer
            invalid_indices = self.sampling_check(pulse_indices)

        input_array = jnp.zeros((self.time_length, self.mod))

        for idx, val in zip(pulse_indices, sequence):
            onehot_vector = jnp.zeros((self.mod,)).at[val].set(self.pulse_amplitude)
            input_array = input_array.at[idx].add(onehot_vector)
            input_array = input_array.at[idx + 1].add(onehot_vector)

        modulo_sum = sequence.sum() % self.mod
        label_value = 1.0 if modulo_sum == self.congruent_number else -1.0
        output_array = jnp.tile(jnp.array([label_value]), (self.time_length, 1))

        return input_array, output_array

    def create_trial_with_indices(self, sequence, pulse_indices):
        """
        Create a single trial with a specific sequence of integers and fixed pulse indices.

        Args:
            sequence (jnp.ndarray): 1D array of length `num_pulses`, integers in [0, M-1].
            pulse_indices (jnp.ndarray): 1D array of length `num_pulses`, sorted time indices
                                        at which to place each pulse.

        Returns:
            tuple:
                - input_array (jnp.ndarray): Shape (time_length, mod), one-hot pulses over time.
                - output_array (jnp.ndarray): Shape (time_length, 1), tiled label (+1 or -1).
        """
        assert len(sequence) == len(
            pulse_indices
        ), "Sequence and indices must be the same length"
        assert not self.sampling_check(
            pulse_indices
        ), "Pulse indices must satisfy gap requirement"

        input_array = jnp.zeros((self.time_length, self.mod))

        for idx, val in zip(pulse_indices, sequence):
            onehot_vector = jnp.zeros((self.mod,)).at[val].set(self.pulse_amplitude)
            input_array = input_array.at[idx].add(onehot_vector)
            input_array = input_array.at[idx + 1].add(onehot_vector)

        modulo_sum = sequence.sum() % self.mod
        label_value = 1.0 if modulo_sum == self.congruent_number else -1.0
        output_array = jnp.tile(jnp.array([label_value]), (self.time_length, 1))

        return input_array, output_array

    def build_balanced_dataset(self):
        """
        Build a balanced dataset of positive and negative trials.

        Half of the trials will be congruent (positive class) and half incongruent (negative class).
        """
        trials_per_label = self.num_trials // 2
        positive_trials = []
        negative_trials = []

        while (
            len(positive_trials) < trials_per_label
            or len(negative_trials) < trials_per_label
        ):
            subkey = self.split_key()
            sequence = random.randint(subkey, (self.num_pulses,), 0, self.mod)
            inputs, targets = self.create_trial(sequence)
            if targets[0] == 1.0 and len(positive_trials) < trials_per_label:
                positive_trials.append((inputs, targets))
            elif targets[0] == -1.0 and len(negative_trials) < trials_per_label:
                negative_trials.append((inputs, targets))

        self.dataset = (
            positive_trials[:trials_per_label] + negative_trials[:trials_per_label]
        )
        assert self.num_trials == len(
            self.dataset
        ), "Dataset length does not agree with desired length"

    def return_number_of_sequences(
        self,
    ):
        """
        Simple function to compute the number of sequences.
        """
        return len(list(product(range(self.mod), repeat=self.num_pulses)))

    def build_sequence_balanced_dataset(self, trials_per_pos_seq, trials_per_neg_seq):
        """
        Builds a dataset via balancing individual sequences in positive and negative trials.
        """
        all_sequences = list(product(range(self.mod), repeat=self.num_pulses))

        pos_sequences = []
        neg_sequences = []

        for seq in all_sequences:
            seq_array = jnp.array(seq)
            modulo_sum = seq_array.sum() % self.mod
            label_value = 1.0 if modulo_sum == self.congruent_number else -1.0
            if label_value == 1.0:
                pos_sequences.append(seq_array)
            else:
                neg_sequences.append(seq_array)

        dataset = []

        # Positive sequences
        for seq in pos_sequences:
            for _ in range(trials_per_pos_seq):
                inputs, targets = self.create_trial(seq)
                dataset.append((inputs, targets))

        # Negative sequences
        for seq in neg_sequences:
            for _ in range(trials_per_neg_seq):
                inputs, targets = self.create_trial(seq)
                dataset.append((inputs, targets))

        self.dataset = dataset

    def generate_jax_tensor(self):
        """
        Convert the dataset to JAX tensors.

        Returns:
            tuple:
                - features_tensor (jnp.ndarray): Shape (num_trials, time_length, mod).
                - labels_tensor (jnp.ndarray): Shape (num_trials, time_length, 1).
        """
        features = []
        labels = []
        for x, y in self.dataset:
            features.append(x)
            labels.append(y)
        features_tensor = jnp.stack(features, axis=0)
        labels_tensor = jnp.stack(labels, axis=0)
        return features_tensor, labels_tensor

    def generate_tf_dataset(self, batch_size):
        """
        Convert the dataset into a TensorFlow Dataset for training.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            tf.data.Dataset: A batched and shuffled dataset ready for training.
        """
        features_tensor, labels_tensor = self.generate_jax_tensor()
        dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
        subkey = self.split_key()
        dataset = dataset.shuffle(
            buffer_size=len(self.dataset),
            reshuffle_each_iteration=True,
            seed=subkey[0].item(),
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(2)
        return dataset

    def decode_integer_inputs(self, input_array):
        """
        Decode a sequence of one-hot vectors into integers or None if no pulse.

        Args:
            input_array (jnp.ndarray): Shape (time_length, mod).

        Returns:
            list: A list of integers (0 to M-1) or None for each time step.
        """
        decoded = []
        for t in range(input_array.shape[0]):
            vec = input_array[t]
            if vec.sum() == 0:
                decoded.append(None)
            else:
                decoded.append(int(jnp.argmax(vec)))
        return decoded
