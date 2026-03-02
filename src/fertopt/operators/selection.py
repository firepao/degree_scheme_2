from __future__ import annotations

import numpy as np


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def dynamic_elite_select_indices(
    objective_values: np.ndarray,
    decision_values: np.ndarray,
    select_size: int,
    generation_index: int,
    max_generations: int,
    alpha0: float,
    beta_decay: float,
    omega_f: float,
    omega_x: float,
    k_neighbors: int,
) -> np.ndarray:
    if objective_values.ndim != 2 or decision_values.ndim != 2:
        raise ValueError("objective_values 与 decision_values 必须是二维")
    if objective_values.shape[0] != decision_values.shape[0]:
        raise ValueError("目标与决策样本数必须一致")

    sample_count = objective_values.shape[0]
    if sample_count == 0:
        return np.array([], dtype=int)

    ranks = _non_dominated_ranks(objective_values)

    omega_f = float(np.clip(omega_f, 0.0, 1.0))
    omega_x = float(np.clip(omega_x, 0.0, 1.0))
    sum_w = max(omega_f + omega_x, 1e-8)
    omega_f /= sum_w
    omega_x /= sum_w

    combined_dist = _combined_distance(objective_values, decision_values, omega_f, omega_x)
    k = int(np.clip(k_neighbors, 1, max(sample_count - 1, 1)))
    sparsity = _local_sparsity(combined_dist, k)

    alpha_t = float(alpha0 * np.exp(-beta_decay * generation_index / max(max_generations, 1)))
    sparsity_norm = sparsity / max(float(np.max(sparsity)), 1e-8)
    scores = 1.0 / np.maximum(ranks.astype(float), 1.0) + alpha_t * sparsity_norm

    selected = np.argsort(-scores)[:select_size]
    return selected.astype(int)


def _combined_distance(
    objective_values: np.ndarray,
    decision_values: np.ndarray,
    omega_f: float,
    omega_x: float,
) -> np.ndarray:
    obj_diff = objective_values[:, None, :] - objective_values[None, :, :]
    dec_diff = decision_values[:, None, :] - decision_values[None, :, :]
    obj_dist = np.linalg.norm(obj_diff, axis=2)
    dec_dist = np.linalg.norm(dec_diff, axis=2)
    return omega_f * obj_dist + omega_x * dec_dist


def _local_sparsity(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    sparsity = np.zeros(n, dtype=float)
    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf
        nearest = np.partition(row, k)[:k]
        sparsity[i] = float(np.mean(nearest))
    return sparsity


def _non_dominated_ranks(objective_values: np.ndarray) -> np.ndarray:
    n = objective_values.shape[0]
    dominates = [set() for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        dominates_set = set()
        for j in range(n):
            if i == j:
                continue
            if _dominates(objective_values[i], objective_values[j]):
                dominates_set.add(j)
            elif _dominates(objective_values[j], objective_values[i]):
                dominated_count[i] += 1
        dominates[i] = dominates_set
        if dominated_count[i] == 0:
            fronts[0].append(i)

    ranks = np.zeros(n, dtype=int)
    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: list[int] = []
        for p in fronts[front_idx]:
            ranks[p] = front_idx + 1
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        front_idx += 1
    return ranks


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)
