from __future__ import annotations

import numpy as np

from fertopt.operators.initialization import beta_biased_initialize


def test_beta_initialization_respects_bounds() -> None:
    rng = np.random.default_rng(123)
    pop = beta_biased_initialize(
        pop_size=40,
        dimension=12,
        lower_bound=0.0,
        upper_bound=300.0,
        strength_k=8.0,
        rng=rng,
    )

    assert pop.shape == (40, 12)
    assert np.min(pop) >= 0.0
    assert np.max(pop) <= 300.0


def test_beta_initialization_shows_bias_trend() -> None:
    rng = np.random.default_rng(123)
    pop = beta_biased_initialize(
        pop_size=60,
        dimension=6,
        lower_bound=0.0,
        upper_bound=1.0,
        strength_k=10.0,
        rng=rng,
    )

    low_group_mean = pop[:10].mean()
    high_group_mean = pop[-10:].mean()
    assert high_group_mean > low_group_mean
