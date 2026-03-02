from __future__ import annotations

import numpy as np

from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, nondominated_mask


def test_nondominated_mask_basic() -> None:
    vals = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    mask = nondominated_mask(vals)
    assert mask.tolist() == [True, True, False]


def test_igd_and_hv_non_negative() -> None:
    solution = np.array([[1.2, 2.1], [2.2, 1.1]], dtype=float)
    reference = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    ref_point = np.array([3.0, 3.0], dtype=float)

    igd_val = igd(solution, reference)
    hv_val = hypervolume_monte_carlo(solution, ref_point=ref_point, samples=5000, seed=1)

    assert igd_val >= 0.0
    assert hv_val >= 0.0
