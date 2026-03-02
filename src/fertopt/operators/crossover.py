from __future__ import annotations

import numpy as np


def build_elite_prototypes(
    population: np.ndarray,
    objective_values: np.ndarray,
    prototype_count: int,
    elite_ratio: float,
    kmeans_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        from deap import base, creator, tools
    except ImportError:
        pass

    if population.ndim != 2:
        raise ValueError("population 必须为二维")
    if objective_values.ndim != 2:
        raise ValueError("objective_values 必须为二维")
    if population.shape[0] != objective_values.shape[0]:
        raise ValueError("population 与 objective_values 样本数必须一致")

    sample_count = population.shape[0]
    elite_num = max(2, int(sample_count * elite_ratio))
    elite_num = min(elite_num, sample_count)

    try:
        from deap import tools, base, creator
        
        # We need a temporary individual class that has fitness attribute
        # We can reuse deap's structure if available, or mock it
        
        # Check if FitnessMin already exists in creator (it should if runner ran)
        # But here we are in a module.
        # Safe way: Create local dummy classes for selection purpose
        
        class LocalFitness(base.Fitness):
            weights = (-1.0,) * objective_values.shape[1]

        class LocalInd(list):
            def __init__(self, values, index):
                self.extend(values)
                self.index = index
                self.fitness = LocalFitness()
                self.fitness.values = tuple(objective_values[index])

        # Create population of LocalInd
        inds = []
        for i in range(sample_count):
            inds.append(LocalInd(population[i], i))

        # Use NSGA-II selection
        # This will select based on rank and crowding distance
        selected = tools.selNSGA2(inds, elite_num)
        elite_indices = [ind.index for ind in selected]
    
    except ImportError:
        # Fallback: Simple Sum of Normalized Objectives
        # Normalize to [0, 1] range first to avoid dominance by large value objectives
        min_vals = np.min(objective_values, axis=0)
        max_vals = np.max(objective_values, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0 # Avoid div by zero
        
        norm_obj = (objective_values - min_vals) / ranges
        score = np.sum(norm_obj, axis=1)
        elite_indices = np.argsort(score)[:elite_num]

    elites = population[elite_indices]

    k = max(1, min(int(prototype_count), elites.shape[0]))
    return _kmeans_numpy(elites, k=k, iters=max(1, int(kmeans_iters)), rng=rng)


def prototype_guided_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    prototypes: np.ndarray,
    stage_count: int,
    gamma0: float,
    generation_index: int,
    max_generations: int,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if stage_count <= 0:
        raise ValueError("stage_count 必须大于 0")

    dim = parent_a.shape[0]
    if dim != parent_b.shape[0]:
        raise ValueError("父代维度必须一致")
    
    # 允许非标准维度的处理，但警告（或根据维度调整 alpha）
    # if dim != stage_count * 3:
    #    raise ValueError("当前实现要求每阶段为 NPK 三维")

    # Gamma Annealing: Linear decay
    anneal = 1.0 - (generation_index / max(1, max_generations))
    gamma = float(np.clip(gamma0 * anneal, 0.0, 1.0))

    # Crossover: Simulated Binary Crossover (SBX) or Arithmetic?
    # Original implementation was Arithmetic with Beta distribution.
    # Let's weaken the arithmetic mixing to be more exploration-friendly?
    # Standard Arithmetic: alpha * P1 + (1-alpha) * P2
    
    # Generate alpha for each stage (or gene?)
    # Generating per gene increases diversity.
    alpha = rng.beta(2.0, 2.0, size=dim)
    
    child_1 = alpha * parent_a + (1.0 - alpha) * parent_b
    child_2 = alpha * parent_b + (1.0 - alpha) * parent_a

    # Prototype Guidance
    # Instead of pulling towards nearest prototype of PARENT (which reinforces local optima),
    # let's pull towards a RANDOM prototype from the elite set to encourage jumping?
    # Or keep nearest but with much smaller probability?
    
    # Strategy: Only apply guidance with probability p_guide
    p_guide = 0.5 
    if rng.random() < p_guide:
        # Select prototype: Nearest or Random?
        # Random prototype maintains diversity better.
        proto_idx_a = rng.integers(0, len(prototypes))
        proto_idx_b = rng.integers(0, len(prototypes))
        proto_a = prototypes[proto_idx_a]
        proto_b = prototypes[proto_idx_b]
        
        # Pull
        child_1 = (1.0 - gamma) * child_1 + gamma * proto_a
        child_2 = (1.0 - gamma) * child_2 + gamma * proto_b

    # Bound
    child_1 = np.clip(child_1, lower_bound, upper_bound)
    child_2 = np.clip(child_2, lower_bound, upper_bound)
    
    return child_1, child_2


def _nearest_prototype(vector: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(prototypes - vector, axis=1)
    return prototypes[int(np.argmin(distance))]


def _kmeans_numpy(data: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    if data.shape[0] <= k:
        return data.copy()

    init_idx = rng.choice(data.shape[0], size=k, replace=False)
    centers = data[init_idx].copy()

    for _ in range(iters):
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()

        for idx in range(k):
            cluster_points = data[labels == idx]
            if cluster_points.shape[0] == 0:
                new_centers[idx] = data[rng.integers(0, data.shape[0])]
            else:
                new_centers[idx] = np.mean(cluster_points, axis=0)

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return centers
