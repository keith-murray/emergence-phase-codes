# Emergence of phase codes in recurrent networks trained to perform modular arithmetic

<div align="center">
<img src="https://github.com/keith-murray/emergence-phase-codes/blob/main/results/pca_animation.gif" alt="PCA Animation" width="350">
</div>

## Abstract
Recurrent neural networks (RNNs) can implement complex computations by leveraging a range of dynamics, such as oscillations, attractors, and transient trajectories. A growing body of work has highlighted the emergence of phase codes—a type of oscillatory activity where information is encoded in the relative phase of network activity—in RNNs trained for working memory tasks. However, these studies rely on architectural constraints or regularization schemes that explicitly promote oscillatory solutions. Here, we investigate whether phase coding can emerge purely from task optimization by training continuous-time RNNs (CT-RNNs) to perform a simple modular arithmetic task without oscillatory-promoting biases. We find that in the absence of such biases, CT-RNNs can learn phase code solutions. Surprisingly, we also uncover a rich diversity of alternative solutions that solve our modular arithmetic task via qualitatively distinct dynamics and dynamical mechanisms. We map the solution space for our task and show that the phase code solution occupies a distinct region. These results suggest that phase coding can be a natural but not inevitable outcome of training RNNs on symbolic tasks, and highlight the heterogeneity of dynamical mechanisms that can solve simple tasks like modular arithmetic.

## Installation
1. Clone the repository:
```
git clone https://github.com/keith-murray/emergence-phase-codes.git
cd emergence-phase-codes
```
2. Install `poetry`:
```
curl -sSL https://install.python-poetry.org | python3 -
```
3. Install packages associated with the repository:
```
poetry install
```
This repository installs my `ctrnn-jax` package from [here](https://github.com/keith-murray/ctrnn-jax).

## Usage
The `notebooks/` directory contains Jupyter Notebooks for making the phase code and solution space figures.

The `scripts/` directory contains python scripts for performing hyperparameter sweeps. To reproduce:
```
poetry run python ./scripts/generate_configs.py
```
and then:
```
poetry run python ./scripts/launch_jobs.py
```
To reproduce the solution space figure:
```
poetry run python ./scripts/dsa_pipeline.py
```
