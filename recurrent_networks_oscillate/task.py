"""Implementation of the `Modulo3Arithmetic` task."""

from itertools import product

import jax.numpy as jnp
from jax import random
import tensorflow as tf


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
class Modulo3Arithmetic:
    """An implementation of the Modulo 3 Arithmetic task."""

    def __init__(
        self,
        key,
        neurons=100,
        time=50,
        min_training_trials=50,
        testing_trials=1,
    ):
        """
        Initialize the Modulo3Arithmetic class.

        Parameters:
            key (random.PRNGKey): The initial random key.
            min_training_trials (int): The minimum number of training trials per integer sequence.
            testing_trials (int): The number of testing trials.
        """
        self.key = key
        self.neurons = neurons
        self.time = time
        self.min_training_trials = min_training_trials
        self.testing_trials = testing_trials

        self.generate_encoded_integers()
        self.instantiate_sequence_dict()

        self.training_dict = self.create_data_dict()
        self.fill_data_dict(
            self.training_dict,
            self.min_training_trials,
        )
        self.balance_data_dict(
            self.training_dict,
            self.min_training_trials,
        )

        self.testing_dict = self.create_data_dict()
        self.fill_data_dict(
            self.testing_dict,
            self.testing_trials,
        )

    def generate_subkey(
        self,
    ):
        """
        Generate a new subkey for random operations.

        Returns:
            random.PRNGKey: The new subkey.
        """
        key, subkey = random.split(self.key)
        self.key = key
        return subkey

    def generate_encoded_integers(
        self,
    ):
        """
        Generate encoded integer vectors for '0', '1', and '2'.

        Returns:
            dict: A dictionary matching integers to encoding vectors.
        """
        encoded_integers = {}

        subkey = self.generate_subkey()
        for integer, new_key in zip(["0", "1", "2"], random.split(subkey, 3)):
            encoded_integers[integer] = random.normal(new_key, (self.neurons,))

        self.encoded_integers = encoded_integers

    def create_data_dict(
        self,
    ):
        """
        Create a dictionary of all integer sequences.

        Returns:
            dict: A dictionary matching integer sequences
                  (e.g. '201', '221', etc.) with empty lists.
        """
        all_sequences = product(["0", "1", "2"], repeat=3)
        data_dict = {"".join(seq): [] for seq in all_sequences}
        return data_dict

    def instantiate_sequence_dict(
        self,
    ):
        """
        Creates and fills the integer_sequences dictionary.
        """
        self.integer_sequences = self.create_data_dict()

        for integer_sequence in self.integer_sequences:
            unique_ints = len(set(list(integer_sequence)))

            if unique_ints == 2:
                self.integer_sequences[integer_sequence] = -1
            else:
                self.integer_sequences[integer_sequence] = 1

    def print_integer_sequences(
        self,
    ):
        """
        Print the contents of integer_sequences with integer sequences
        in the first column and the label in the second column.
        """
        print("Contents of integer_sequences dictionary:")
        print("Integer sequence | Label")
        for sequence, label in self.integer_sequences.items():
            print(f"{sequence} | {label}")

    def create_trial(
        self,
        integer_sequence,
        unique_integers,
    ):
        """
        Create a trial with random "ball" positions and associated integers.

        Parameters:
            integer_sequence (str): A string containing a sequence of integers like
                                    '221', '001', etc.
            unique_integers (int): The number of unique colors in the combination.

        Returns:
            tuple:
                - input_array (jnp.ndarray): JAX array of shape (time, neurons)
                                             representing a trial.
                - output_array (jnp.ndarray): JAX array of shape (1), value is
                                              either -1 or 1 based on unique_integers.
        """
        # Initialize a zero JAX array with shape (time, neurons)
        input_array = jnp.zeros((self.time, self.neurons))

        # Select random "ball" positions, ensuring minimum index distance of 5 between each "ball"
        while True:
            subkey = self.generate_subkey()
            ball_indices = sorted(
                random.randint(subkey, shape=(3,), minval=3, maxval=40)
            )
            if all(
                ball_indices[i] - ball_indices[i - 1] >= 5
                for i in range(1, len(ball_indices))
            ):
                break

        # Place the color vectors based on the random "ball" positions
        for i, ball_idx in enumerate(ball_indices):
            color = integer_sequence[i]
            color_vector = self.encoded_integers[color]
            input_array = input_array.at[ball_idx].add(color_vector)
            input_array = input_array.at[ball_idx + 1].add(color_vector)

        # Determine the output_array based on the uniqueness of colors
        output_array = jnp.array([-1.0]) if unique_integers == 2 else jnp.array([1.0])

        return (input_array, output_array)

    def fill_data_dict(
        self,
        data_dict,
        trials,
    ):
        """
        Fill the data dictionary with trials of (input_array, output_array).

        Parameters:
            data_dict (dict): The dictionary to be filled (training or testing).
            trials (int): The number of trials per integer sequence.
        """
        for integer_sequence in data_dict:
            unique_colors = len(set(list(integer_sequence)))
            data_dict[integer_sequence] = [
                self.create_trial(integer_sequence, unique_colors)
                for _ in range(trials)
            ]

    def check_and_return_label(
        self,
        data_dict,
        integer_sequence,
    ):
        """
        Check if a label for a given integer sequence is as expected across all trials.

        Parameters:
            data_dict (dict): The data dictionary containing trials.
            integer_sequence (str): The integer sequence to be checked.

        Returns:
            int: The label of the integer_sequence (-1 or 1).
        """
        trials = data_dict[integer_sequence]
        labels = [trial[1].item() for trial in trials]

        # Check if all labels are the same
        dict_label = self.integer_sequences[integer_sequence]
        assert all(
            label == dict_label for label in labels
        ), "Labels are not the same for all trials."

        return dict_label

    def balance_data_dict(
        self,
        data_dict,
        min_trials,
    ):
        """
        Balances the number of positive and negative trials in the data_dict.

        Parameters:
            data_dict (dict): The data dictionary containing trials.
            min_trials (int): The minimum number of trials per integer sequence.
        """
        positive_sequences = []
        negative_sequences = []

        for sequence in data_dict:
            label = self.check_and_return_label(data_dict, sequence)

            if label == 1:
                positive_sequences.append(sequence)
            elif label == -1:
                negative_sequences.append(sequence)

        if len(positive_sequences) > len(negative_sequences):
            over_represented_sequences = positive_sequences
            over_represented_unique_intergers = 1
            under_represented_sequences = negative_sequences
            under_represented_unique_intergers = 2
        else:
            over_represented_sequences = negative_sequences
            over_represented_unique_intergers = 2
            under_represented_sequences = positive_sequences
            under_represented_unique_intergers = 1

        under_represented_trials = round(
            (len(over_represented_sequences) * min_trials)
            / len(under_represented_sequences)
        )

        for sequence in under_represented_sequences:
            data_dict[sequence] = [
                self.create_trial(sequence, under_represented_unique_intergers)
                for _ in range(under_represented_trials)
            ]

        for sequence in over_represented_sequences:
            if len(data_dict[sequence]) != min_trials:
                data_dict[sequence] = [
                    self.create_trial(sequence, over_represented_unique_intergers)
                    for _ in range(min_trials)
                ]

    def print_data_dict(self, data_dict):
        """
        Print two grids. One grid has all the positive label integer sequences,
        and the other grid has all the negative label integer sequences.

        Parameters:
            data_dict (dict): The data dictionary containing trials.
        """
        print("Accepting Grid:")
        print("Integer Sequence | Number of Trials | Status")
        for seq in data_dict:
            label = self.check_and_return_label(data_dict, seq)
            if label == 1:
                print(f"{seq} | {len(data_dict[seq])}")

        print("\nRejecting Grid:")
        print("Integer Sequence | Number of Trials | Status")
        for seq in data_dict:
            label = self.check_and_return_label(data_dict, seq)
            if label == -1:
                print(f"{seq} | {len(data_dict[seq])}")

    def print_training_testing(
        self,
    ):
        """
        Print all relevant information pertaining to training and testing dicts.
        """
        print("\nTRAINING DATA\n")
        self.print_data_dict(self.training_dict)
        print("\n----------")
        print("\nTESTING DATA\n")
        self.print_data_dict(self.testing_dict)
        print("\n")

    def generate_jax_tensor(self, data_dict):
        """
        Create features and labels tensors for the tensorflow dataset.

        Parameters:
            data_dict (dict): The data dictionary of integer sequences and trials.

        Returns:
            tuple:
                - features_tensor (jnp.ndarray): Feature tensor of shape
                                                 (total_trials, time, neurons).
                - labels_tensor (jnp.ndarray): Label tensor of shape (total_trials, 5, 1).
        """
        # Initialize lists to hold feature and label arrays
        features_list = []
        labels_list = []

        # Iterate through each SET_combination in the data_dict
        for _, trials in data_dict.items():
            for input_array, output_array in trials:
                features_list.append(input_array)

                # Expand the label from shape (1) to (5, 1)
                expanded_label = jnp.tile(
                    output_array, (5, 1)
                )  # Could use (self.time, 1)
                labels_list.append(expanded_label)

        # Stack the features and labels to create the final tensors
        features_tensor = jnp.stack(features_list, axis=0)
        labels_tensor = jnp.stack(labels_list, axis=0)

        return features_tensor, labels_tensor

    def generate_tf_dataset(self, data_dict, batch_size):
        """
        Create a TensorFlow Dataset object from the provided data_dict and batch size.

        This method first generates JAX tensors for the features and labels using the
        `generate_jax_tensor` method. These tensors are then converted into a TensorFlow
        Dataset, which is shuffled and batched based on the provided batch size.

        Parameters:
            data_dict (dict): The data dictionary containing trials for each integer sequence.
            batch_size (int): The number of samples to include in each batch.

        Returns:
            tf.data.Dataset: A shuffled and batched TensorFlow Dataset object.
        """
        features_tensor, labels_tensor = self.generate_jax_tensor(data_dict)
        dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
        subkey = self.generate_subkey()
        dataset = dataset.shuffle(
            dataset.cardinality(), reshuffle_each_iteration=True, seed=subkey[0].item()
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(2)

        return dataset

    def tf_datasets(self, train_batch_size):
        """
        Create both training and testing TensorFlow Datasets.

        Returns:
            training_tf_dataset (tf.data.Dataset): A shuffled and batched training dataset.
            testing_tf_dataset (tf.data.Dataset): A shuffled and batched testing dataset.
        """
        training_tf_dataset = self.generate_tf_dataset(
            self.training_dict, train_batch_size
        )
        test_batch_size = sum(len(trials) for _, trials in self.testing_dict.items())
        testing_tf_dataset = self.generate_tf_dataset(
            self.testing_dict, test_batch_size
        )

        return training_tf_dataset, testing_tf_dataset
