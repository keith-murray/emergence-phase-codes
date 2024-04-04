import jax.numpy as jnp
from jax import random
from sklearn.decomposition import PCA

def compute_cumulative_variance(model, params, dataset, key, save_loc):
    rates_list = []

    for inputs, _ in dataset.as_numpy_iterator():
        key, subkey = random.split(key)
        _, rates = model.apply(params, inputs, init_key=subkey,)
        rates_list.append(rates)

    rates = jnp.concatenate(rates_list, axis=0)
    rates_reshaped = rates.reshape(-1, rates.shape[-1])

    pca_full = PCA()
    pca_full.fit(rates_reshaped)
    cumulative_variance = jnp.cumsum(pca_full.explained_variance_ratio_)

    jnp.save(save_loc, cumulative_variance)