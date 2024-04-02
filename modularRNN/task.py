import jax
import jax.numpy as jnp
from jax import random
import tensorflow as tf

class ModularArithmeticTask:
    def __init__(self, key, training_trials, testing_trials, train_batch_size, mod_set, trial_distribution,):
        """
        Initialize the ModularArithmeticTask class to instantiate tasks.
        
        Parameters:
            key (random.PRNGKey): The initial random key.
            training_trials (int): The number of training trials.
            testing_trials (int): The number of testing trials.
            train_batch_size (int): The size of training batches.
            mod_set (arr): The possible modular values for the task.
            trial_distribution (function): The distribution of pulses per trial. The input is a PRNGKey.
        """
        self.key = key
        self.training_trials = training_trials
        self.testing_trials = testing_trials
        self.train_batch_size = train_batch_size
        self.mod_set = mod_set
        self.trial_distribution = trial_distribution
        
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

        Returns:
            boolean
        """
        pass