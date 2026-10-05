"""Multiple testing corrections (Benjamini-Hochberg FDR)."""

import numpy as np


def benjamini_hochberg(p_values, alpha=0.05):
    p = np.array(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not below.any():
        return [], []
    max_k = np.max(np.where(below)[0])
    significant = order[:max_k + 1]
    return list(significant), list(thresholds)


def bh_summary(test_names, p_values, alpha=0.05):
    sig, thresholds = benjamini_hochberg(p_values, alpha)
    print(f"Benjamini-Hochberg FDR (alpha={alpha}):")
    for i, (name, p) in enumerate(zip(test_names, p_values)):
        status = "SIGNIFICANT" if i in sig else "not significant"
        print(f"  {name}: p={p:.4f} [{status}]")
    return set(sig)
