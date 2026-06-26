# Positional Advantage Under Industrial Selection in Scientific Networks

Code for the Bayesian agent-based simulations and figure generation reported in the accompanying paper.

## Overview

This repository contains the simulation code used to investigate how the network position of an industrially selected researcher affects group credences in epistemic communities. The model builds on Holman and Bruner's (2017) industrial selection framework and uses a Moran process to model generational mentorship dynamics across 10 network topologies.

## Files

- `experiment_script.py` — Main simulation engine. Runs all experimental phases (baseline position comparison, symmetric controls, Moran ablation, temporal dynamics), performs statistical analysis, and generates results.
- `figure_generation.py` — Generates the figures reported in the paper.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- NetworkX
- SciPy
- Matplotlib
- tqdm
- pingouin

Install all dependencies with:

```bash
pip install numpy pandas networkx scipy matplotlib tqdm pingouin
```

## Usage

Run the full experiment:

```bash
python experiment_script.py
```

Generate figures:

```bash
python figure_generation.py
```

## References

Holman, B., & Bruner, J. (2017). Experimentation by industrial selection. *Philosophy of Science*, 84(5), 1008–1019.
