from __future__ import annotations

import numpy as np


def nondominated_mask(objective_values: np.ndarray) -> np.ndarray:
    values = np.asarray(objective_values, dtype=float)
    n = values.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            mask[i] = False
    return mask


def igd(solution_set: np.ndarray, reference_front: np.ndarray) -> float:
    s = np.asarray(solution_set, dtype=float)
    r = np.asarray(reference_front, dtype=float)
    if s.size == 0 or r.size == 0:
        return float("nan")
    distances = np.linalg.norm(r[:, None, :] - s[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest))


def hypervolume_monte_carlo(
    objective_values: np.ndarray,
    ref_point: np.ndarray,
    samples: int = 20000,
    seed: int = 42,
) -> float:
    values = np.asarray(objective_values, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if values.size == 0:
        return 0.0

    nd_vals = values[nondominated_mask(values)]
    lower = np.min(nd_vals, axis=0)
    upper = ref
    if np.any(upper <= lower):
        upper = np.maximum(upper, lower + 1e-8)

    rng = np.random.default_rng(seed)
    points = rng.uniform(lower, upper, size=(samples, nd_vals.shape[1]))

    dominated = np.zeros(samples, dtype=bool)
    for p_idx, point in enumerate(points):
        dominated[p_idx] = np.any(np.all(nd_vals <= point, axis=1))

    box_volume = float(np.prod(upper - lower))
    hv = box_volume * float(np.mean(dominated))
    return hv
