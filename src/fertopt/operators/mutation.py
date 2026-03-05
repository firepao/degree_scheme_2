from __future__ import annotations

import numpy as np


def build_synergy_antagonism_matrix(rho_np: float, rho_nk: float, rho_pk: float) -> np.ndarray:
    matrix = np.array(
        [
            [0.0, rho_np, rho_nk],
            [rho_np, 0.0, rho_pk],
            [rho_nk, rho_pk, 0.0],
        ],
        dtype=float,
    )
    return matrix


def dynamic_mutation_probability(
    p0: float,
    beta: float,
    gamma: float,
    delta: float,
    deficiency_score: float,
    stage_sensitivity: float,
    diversity_pressure: float,
    p_max: float,
) -> float:
    raw_p = p0 * (
        1.0
        + beta * deficiency_score
        + gamma * stage_sensitivity
        + delta * diversity_pressure
    )
    return float(np.clip(raw_p, 0.0, p_max))


def coupled_mutation(
    individual: np.ndarray,
    num_stages: int,
    matrix_m: np.ndarray,
    sigma_base: float,
    alpha_knowledge: float,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if individual.ndim != 1:
        raise ValueError("individual 必须为一维向量")
    if individual.shape[0] != num_stages * 3:
        raise ValueError("individual 维度必须为 num_stages * 3")

    out = individual.copy().astype(float)
    span = max(upper_bound - lower_bound, 1e-8)

    for stage_idx in range(num_stages):
        start = stage_idx * 3
        end = start + 3
        z_t = out[start:end]

        scale_vec = np.clip((z_t - lower_bound) / span, 0.0, 1.0) + 1e-3
        d_t = np.diag(scale_vec)
        cov = (sigma_base ** 2) * np.eye(3) + alpha_knowledge * d_t @ matrix_m @ d_t
        cov = 0.5 * (cov + cov.T)
        cov = _make_psd(cov)

        delta = rng.multivariate_normal(mean=np.zeros(3), cov=cov)
        out[start:end] = z_t + delta

    return np.clip(out, lower_bound, upper_bound)


def _make_psd(cov: np.ndarray) -> np.ndarray:
    jitter = 1e-8
    for _ in range(8):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            cov = cov + np.eye(cov.shape[0]) * jitter
            jitter *= 10
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
