from __future__ import annotations

import numpy as np

from fertopt.operators.mutation import (
    build_synergy_antagonism_matrix,
    coupled_mutation,
    dynamic_mutation_probability,
)


def test_dynamic_mutation_probability_is_clipped() -> None:
    p = dynamic_mutation_probability(
        p0=0.1,
        beta=0.8,
        gamma=0.7,
        delta=0.6,
        deficiency_score=1.0,
        stage_sensitivity=1.2,
        diversity_pressure=1.0,
        p_max=0.55,
    )
    assert 0.0 <= p <= 0.55


def test_coupled_mutation_respects_bounds() -> None:
    rng = np.random.default_rng(123)
    matrix = build_synergy_antagonism_matrix(rho_np=0.4, rho_nk=-0.2, rho_pk=0.15)
    individual = np.array([120.0, 90.0, 80.0, 150.0, 140.0, 110.0], dtype=float)

    mutated = coupled_mutation(
        individual=individual,
        num_stages=2,
        matrix_m=matrix,
        sigma_base=2.0,
        alpha_knowledge=0.5,
        lower_bound=0.0,
        upper_bound=300.0,
        rng=rng,
    )

    assert mutated.shape == individual.shape
    assert np.all(mutated >= 0.0)
    assert np.all(mutated <= 300.0)
    assert not np.allclose(mutated, individual)
