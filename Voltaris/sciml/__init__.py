"""Scientific-ML variants for PyBaMM-Conf 2026 comparison.

Contents:
- data.py:    load CALB canonical cells, compute features, K-split
- models.py:  StandardPINN, CausalPINN, OperatorAugmentedPINN
- physics.py: PyBaMM rxn-lim SEI analytical rate + per-cell k_SEI fit
- train.py:   training loop shared by all variants
- evaluate.py: hold-out RMSE metric matching Voltaris/outputs/holdout_sweep/
"""
