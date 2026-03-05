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

    # Calculate Alpha_t (Dynamic Balance Coefficient)
    # alpha_t starts at alpha0 and decays to 0
    alpha_t = float(alpha0 * np.exp(-beta_decay * generation_index / max(max_generations, 1)))
    
    selected_indices = []
    current_rank = 1
    max_rank = int(np.max(ranks)) if len(ranks) > 0 else 0
    
    while len(selected_indices) < select_size and current_rank <= max_rank:
        candidates = np.where(ranks == current_rank)[0]
        
        if len(candidates) == 0:
            current_rank += 1
            continue
            
        n_needed = select_size - len(selected_indices)
        
        if len(candidates) <= n_needed:
            # Entire front fits
            selected_indices.extend(candidates)
        else:
            # Truncation needed: Select based on Crowding Distance + Decision Sparsity
            
            sub_obj = objective_values[candidates]
            sub_dec = decision_values[candidates]
            
            # 1. Crowding Distance (Objective Space) - Standard NSGA-II
            crowding_dist = _crowding_distance(sub_obj)
            
            # 2. Decision Space Sparsity
            # Normalize decision values first
            dec_min = np.min(sub_dec, axis=0)
            dec_max = np.max(sub_dec, axis=0)
            dec_range = dec_max - dec_min
            dec_range[dec_range == 0] = 1.0
            norm_sub_dec = (sub_dec - dec_min) / dec_range
            
            # Calculate Decision Distance Matrix
            dec_dist_matrix = _distance_matrix(norm_sub_dec)
            
            # Calculate Sparsity (Mean distance to k neighbors)
            sub_k = int(np.clip(k_neighbors, 1, max(len(candidates) - 1, 1)))
            sparsity_dec = _local_sparsity(dec_dist_matrix, sub_k)
            
            # 3. Combine Scores
            # Handle Infinity in Crowding Distance (Boundary Points)
            finite_mask = np.isfinite(crowding_dist)
            
            combined_score = np.zeros_like(crowding_dist)
            
            # Boundary points (inf CD) always get infinite score
            combined_score[~finite_mask] = np.inf
            
            # For finite points, normalize and combine
            if np.any(finite_mask):
                # Normalize Finite CD to [0, 1]
                cd_vals = crowding_dist[finite_mask]
                cd_min, cd_max = np.min(cd_vals), np.max(cd_vals)
                cd_range = cd_max - cd_min if cd_max > cd_min else 1.0
                norm_cd = (cd_vals - cd_min) / cd_range
                
                # Normalize Sparsity to [0, 1]
                sp_vals = sparsity_dec[finite_mask]
                sp_min, sp_max = np.min(sp_vals), np.max(sp_vals)
                sp_range = sp_max - sp_min if sp_max > sp_min else 1.0
                norm_sp = (sp_vals - sp_min) / sp_range
                
                # Weighted Sum
                # Note: We want to MAXIMIZE both CD and Sparsity
                combined_score[finite_mask] = (1.0 - alpha_t) * norm_cd + alpha_t * norm_sp
            
            # Select best
            sorted_local_idx = np.argsort(-combined_score) # Descending
            best_local_idx = sorted_local_idx[:n_needed]
            
            best_global_candidates = candidates[best_local_idx]
            selected_indices.extend(best_global_candidates)
            
        current_rank += 1
        
    return np.array(selected_indices, dtype=int)


def _crowding_distance(objective_values: np.ndarray) -> np.ndarray:
    n_points, n_obj = objective_values.shape
    if n_points == 0:
        return np.array([])
    
    distances = np.zeros(n_points)
    
    for i in range(n_obj):
        # Sort by objective i
        sorted_indices = np.argsort(objective_values[:, i])
        obj_min = objective_values[sorted_indices[0], i]
        obj_max = objective_values[sorted_indices[-1], i]
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        if obj_max - obj_min <= 1e-9:
            continue
            
        norm_factor = obj_max - obj_min
        
        # Add normalized distance for intermediate points
        # dist[j] = (val[j+1] - val[j-1]) / (max - min)
        # Vectorized approach for speed
        prev_vals = objective_values[sorted_indices[:-2], i]
        next_vals = objective_values[sorted_indices[2:], i]
        diffs = (next_vals - prev_vals) / norm_factor
        
        distances[sorted_indices[1:-1]] += diffs
            
    return distances


def _distance_matrix(values: np.ndarray) -> np.ndarray:
    # Efficient Euclidean distance matrix
    # (x-y)^2 = x^2 + y^2 - 2xy
    # But for numerical stability in small distances, standard norm is safer given n is small (pop_size ~100)
    diff = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _local_sparsity(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    sparsity = np.zeros(n, dtype=float)
    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf
        # Find k nearest neighbors
        if k >= n - 1:
            # If k is effectively all neighbors
            nearest = row[row < np.inf]
        else:
            nearest = np.partition(row, k)[:k]
            
        if len(nearest) > 0:
            sparsity[i] = float(np.mean(nearest))
        else:
            sparsity[i] = 0.0
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
