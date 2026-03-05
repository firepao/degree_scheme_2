from __future__ import annotations

import numpy as np

from fertopt.operators.crossover import build_elite_prototypes, prototype_guided_crossover


def test_build_elite_prototypes_shape() -> None:
    rng = np.random.default_rng(7)
    population = rng.uniform(0.0, 300.0, size=(30, 12))
    objectives = rng.uniform(0.0, 10.0, size=(30, 3))

    prototypes = build_elite_prototypes(
        population=population,
        objective_values=objectives,
        prototype_count=4,
        elite_ratio=0.3,
        kmeans_iters=10,
        rng=rng,
    )

    assert prototypes.shape == (4, 12)


def test_prototype_guided_crossover_bounds_and_attraction() -> None:
    rng = np.random.default_rng(11)
    parent_a = np.array([130.0, 150.0, 140.0, 145.0, 160.0, 155.0], dtype=float)
    parent_b = np.array([220.0, 180.0, 210.0, 200.0, 170.0, 190.0], dtype=float)
    prototypes = np.array(
        [
            [40.0, 55.0, 45.0, 50.0, 65.0, 60.0],
            [210.0, 175.0, 200.0, 195.0, 165.0, 185.0],
        ],
        dtype=float,
    )

    child1, child2 = prototype_guided_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        prototypes=prototypes,
        stage_count=2,
        gamma0=1.0,
        generation_index=0,
        max_generations=20,
        lower_bound=0.0,
        upper_bound=300.0,
        rng=rng,
    )

    assert child1.shape == parent_a.shape
    assert child2.shape == parent_a.shape
    assert np.all(child1 >= 0.0) and np.all(child1 <= 300.0)
    assert np.all(child2 >= 0.0) and np.all(child2 <= 300.0)
    nearest_a_idx = int(np.argmin(np.linalg.norm(prototypes - parent_a, axis=1)))
    nearest_b_idx = int(np.argmin(np.linalg.norm(prototypes - parent_b, axis=1)))
    assert np.allclose(child1, prototypes[nearest_a_idx])
    assert np.allclose(child2, prototypes[nearest_b_idx])
