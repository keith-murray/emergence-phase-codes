import jax.numpy as jnp
from jax import random
from jax.nn import one_hot
from tqdm import tqdm
import tensorflow as tf

class ModularArithmeticTask:
    def __init__(self, key, training_trials, testing_trials, train_batch_size, mod_set, pulse_distribution,):
        """
        Initialize the ModularArithmeticTask class to instantiate tasks.
        
        Parameters:
            key (random.PRNGKey): The initial random key.
            training_trials (int): The number of training trials.
            testing_trials (int): The number of testing trials.
            train_batch_size (int): The size of training batches.
            mod_set (arr): The possible modular values for the task.
            pulse_distribution (function): The distribution of pulses per trial. The input is a PRNGKey.
        """
        self.key = key
        self.training_trials = training_trials
        self.testing_trials = testing_trials
        self.train_batch_size = train_batch_size
        self.mod_set = mod_set
        self.pulse_distribution = pulse_distribution
        
    def generate_subkey(self,):
        """
        Generate a new subkey for random operations.
        
        Returns:
            random.PRNGKey: The new subkey.
        """
        key, subkey = random.split(self.key)
        self.key = key
        return subkey
    
    def sample_modular_value(self,):
        """
        Uniformly sample a modular value from self.mod_set

        Returns:
            int: A modular value
        """
        return random.permutation(self.generate_subkey(), self.mod_set)[0]
    
    def test_pulse_indicies(self, pulse_indicies):
        """
        Test an array of pulse indicies to determine if gaps between indicies are greater than 2 indicies.
        Assume that pulse_indicies is sorted.

        Returns:
            boolean
        """
        three_gap = jnp.all(jnp.diff(pulse_indicies) >= 2) # 2 could be config
        # Keep in mind that the differences between pulses can cause slow downs in generation of trials
        greater_than_two = pulse_indicies[0] > 2 # 2 could be config
        less_than_ninety_seven = pulse_indicies[-1] < 97 # 97 could be config

        return three_gap and greater_than_two and less_than_ninety_seven
    
    def generate_pulse_indicies(self, pulse_amount):
        """
        Generate a valid array of sorted pulse indicies.

        Returns:
            pulse_indicies (arr)
        """
        # valid_pulses = False

        # while not valid_pulses:
            # Could modify 100 to change trial length
            # pulse_indicies = random.permutation(self.generate_subkey(), 100)[:pulse_amount].sort()
            # valid_pulses = self.test_pulse_indicies(pulse_indicies)
        pulse_indicies = random.permutation(self.generate_subkey(), 100)[:pulse_amount].sort()
        return pulse_indicies
    
    def generate_pulse_values(self, pulse_amount, mod_value):
        """
        Generate an array of pulse values drawn from [1, mod_value).

        Returns:
            pulse_values (arr)
        """
        return random.randint(self.generate_subkey(), (pulse_amount,), 1, mod_value)
    
    def create_pulses_and_cumulative_mod(self, pulse_indicies, pulse_values, mod_value):
        """
        Create the pulses and cumulative mod array from the previously generated
        pulse_indicies and pulse_values.

        Returns:
            pulses (arr)
            cumulative_mod (arr)
        """
        pulses = jnp.zeros(100)
        pulses = pulses.at[pulse_indicies].set(pulse_values)
        cumulative_sum = jnp.cumsum(pulses)
        cumulative_mod = cumulative_sum % mod_value
        # Strange to think that this one line constitutes the majority of the dataset's computation

        return jnp.asarray(pulses, dtype=jnp.int32), jnp.asarray(cumulative_mod, dtype=jnp.int32)
    
    def create_input_output_tensors(self, pulses, cumulative_mod, mod_value):
        """
        Create input and output tensors from previously defined pulses.

        Returns:
            input_tensor (arr)
            output_tensor (arr)
        """
        mod_value_array = jnp.asarray(mod_value * jnp.ones(100), dtype=jnp.int32) # Again 100 could be variable
        mod_value_tensor = one_hot(mod_value_array, 12)[:, 2:]

        pulses_tensor = one_hot(pulses, 11)[:, 1:]
        input_tensor = jnp.concatenate((mod_value_tensor, pulses_tensor), axis=1)

        output_tensor = one_hot(cumulative_mod, 10)

        return input_tensor, output_tensor
    
    def generate_task_trial(self,):
        """
        Generate an instance of the task, a trial.

        Returns:
            input_tensor (arr)
            output_tensor (arr)
        """
        pulse_amount = self.pulse_distribution(self.generate_subkey())
        mod_value = self.sample_modular_value()
        pulse_indicies = self.generate_pulse_indicies(pulse_amount)
        pulse_values = self.generate_pulse_values(pulse_amount, mod_value)
        pulses, cumulative_mod = self.create_pulses_and_cumulative_mod(pulse_indicies, pulse_values, mod_value)
        input_tensor, output_tensor = self.create_input_output_tensors(pulses, cumulative_mod, mod_value)

        return input_tensor, output_tensor
    
    def generate_trials(self, trial_amount):
        """
        Generate trials ...
        """
        inputs_tensor = []
        outputs_tensor = []

        for i in tqdm(range(trial_amount)):
            input_tensor, output_tensor = self.generate_task_trial()
            inputs_tensor.append(input_tensor)
            outputs_tensor.append(output_tensor)

        inputs_tensor = jnp.stack(inputs_tensor, axis=0)
        outputs_tensor = jnp.stack(outputs_tensor, axis=0)

        return inputs_tensor, outputs_tensor
    
    def create_tf_dataset(self, trial_amount, batch_size):
        """
        Create TensorFlow dataset ...
        """
        inputs_tensor, outputs_tensor = self.generate_trials(trial_amount)
        tf_dataset = tf.data.Dataset.from_tensor_slices((inputs_tensor, outputs_tensor))
        tf_dataset = tf_dataset.shuffle(
            tf_dataset.cardinality(), 
            reshuffle_each_iteration=True, 
            seed=self.generate_subkey()[0].item()
        )
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(2)

        return tf_dataset
    
    def tf_datasets(self,):
        """
        Create TensorFlow datasets for both training and testing.
        """
        training_dataset = self.create_tf_dataset(self.training_trials, self.train_batch_size)
        testing_dataset = self.create_tf_dataset(self.testing_trials, self.testing_trials)

        return training_dataset, testing_dataset