from __future__ import annotations

import numpy as np

from fertopt.operators.selection import dynamic_elite_select_indices


def test_dynamic_elite_select_indices_shape_and_unique() -> None:
    rng = np.random.default_rng(202)
    obj = rng.uniform(0.0, 10.0, size=(40, 3))
    dec = rng.uniform(0.0, 300.0, size=(40, 12))

    indices = dynamic_elite_select_indices(
        objective_values=obj,
        decision_values=dec,
        select_size=20,
        generation_index=5,
        max_generations=60,
        alpha0=0.5,
        beta_decay=3.0,
        omega_f=0.6,
        omega_x=0.4,
        k_neighbors=5,
    )

    assert indices.shape == (20,)
    assert len(set(indices.tolist())) == 20
    assert np.all(indices >= 0)
    assert np.all(indices < 40)
