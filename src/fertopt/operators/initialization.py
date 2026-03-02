from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def beta_biased_initialize(
    pop_size: int,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    strength_k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """连续型偏向随机初始化。

    对第 i 个个体采用:
      s_i = (i - 0.5) / N
      alpha = 1 + K * s_i
      beta  = 1 + K * (1 - s_i)
    """
    if pop_size <= 0:
        raise ValueError("pop_size 必须大于 0")
    if dimension <= 0:
        raise ValueError("dimension 必须大于 0")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound 必须大于 lower_bound")

    population = np.zeros((pop_size, dimension), dtype=float)
    for i in range(1, pop_size + 1):
        s_i = (i - 0.5) / pop_size
        alpha = 1.0 + strength_k * s_i
        beta = 1.0 + strength_k * (1.0 - s_i)
        r = rng.beta(alpha, beta, size=dimension)
        population[i - 1, :] = lower_bound + r * (upper_bound - lower_bound)

    return population


def save_init_distribution_plot(population: np.ndarray, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    flattened = population.ravel()
    plt.figure(figsize=(8, 4))
    plt.hist(flattened, bins=30, color="#4C72B0", alpha=0.85)
    plt.title("Beta-biased Initialization Distribution")
    plt.xlabel("Fertilizer value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
