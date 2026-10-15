# Positional Advantage Under Industrial Selection in Scientific Networks

Bayesian agent-based simulations investigating how network position moderates the effectiveness of industrial selection in epistemic communities.

## Files

- `experiment_script.py` — Main simulation engine (baseline position comparison, symmetric controls, Moran ablation, temporal dynamics)
- `modified-experiment.py` — Robustness checks: stripped-down model variants (Section 5.5 and Appendix A)
- `figure_generation.py` — Figure generation

## Dependencies

Python 3.8+ with: NumPy, Pandas, NetworkX, SciPy, Matplotlib, tqdm, pingouin

```bash
pip install numpy pandas networkx scipy matplotlib tqdm pingouin
```

## Usage

```bash
python experiment_script.py
python modified-experiment.py
python figure_generation.py
```

## References

Holman, B., & Bruner, J. (2017). Experimentation by industrial selection. *Philosophy of Science*, 84(5), 1008–1019.
