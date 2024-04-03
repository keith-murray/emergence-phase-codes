import jax
import jax.numpy as jnp
from jax import random
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
        three_gap = jnp.all(jnp.diff(pulse_indicies) >= 3) # 3 could be config
        greater_than_two = pulse_indicies[0] > 2 # 2 could be config
        less_than_ninety_seven = pulse_indicies[-1] < 97 # 97 could be config

        return three_gap and greater_than_two and less_than_ninety_seven
    
    def generate_pulse_indicies(self, pulse_amount):
        """
        Generate a valid array of sorted pulse indicies.

        Returns:
            pulse_indicies (arr)
        """
        valid_pulses = False

        while not valid_pulses:
            # Could modify 100 to change trial length
            pulse_indicies = random.permutation(self.generate_subkey(), 100)[:pulse_amount].sort()
            valid_pulses = self.test_pulse_indicies(pulse_indicies)
        
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

        return pulses, cumulative_mod