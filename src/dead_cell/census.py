"""Andar ka kabristan: dead rows, dead columns, never-active neurons.

Trained models often carry structure that is never used again:
weight rows that are exactly zero, columns that receive nothing,
and neurons that never fire above a threshold on any sample.
DEAD-CELL maps that waste. Pure NumPy.
"""

import numpy as np


def weight_report(weight, name=None):
    """Zero rows and zero columns in one weight matrix."""
    w = np.asarray(weight, dtype=float)
    if w.ndim != 2:
        raise ValueError("weight must be a 2-D matrix")
    zero_rows = [int(i) for i in np.where(np.all(w == 0.0, axis=1))[0]]
    zero_cols = [int(j) for j in np.where(np.all(w == 0.0, axis=0))[0]]
    zero_params = int(np.sum(w == 0.0))
    return {
        "name": name,
        "shape": [int(w.shape[0]), int(w.shape[1])],
        "params": int(w.size),
        "zero_params": zero_params,
        "zero_rows": zero_rows,
        "zero_cols": zero_cols,
        "waste_fraction": zero_params / w.size,
    }


def dead_cells(weights, activations=None, act_threshold=1e-6):
    """Per-layer waste map, optionally using activation snapshots."""
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise ValueError("weights must be a non-empty list")
    if activations is not None and len(activations) != len(weights):
        raise ValueError("activations must align with weights")
    entries = []
    for i, w in enumerate(weights):
        arr = np.asarray(w, dtype=float)
        if arr.ndim != 2:
            raise ValueError("each weight must be a 2-D matrix")
        entry = weight_report(arr, name="layer_" + str(i))
        if activations is not None:
            a = np.asarray(activations[i], dtype=float)
            if a.ndim != 2 or a.shape[1] != arr.shape[0]:
                raise ValueError("activations must have shape (samples, neurons) aligned with weight rows")
            active = np.any(a > act_threshold, axis=0)
            entry["never_active_neurons"] = [int(j) for j in np.where(~active)[0]]
            entry["occupancy_rate"] = float(np.mean(active))
        entries.append(entry)
    return entries


def total_waste(entries):
    """Aggregate waste across all layers of a dead_cells report."""
    if not isinstance(entries, (list, tuple)) or len(entries) == 0:
        raise ValueError("entries must be a non-empty list of weight_report dicts")
    total_params = sum(e["params"] for e in entries)
    total_zero = sum(e["zero_params"] for e in entries)
    return {
        "num_layers": len(entries),
        "total_params": total_params,
        "total_zero_params": total_zero,
        "waste_fraction": total_zero / total_params if total_params else 0.0,
    }