from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AppConfig


@dataclass(slots=True)
class FertilizationProblem:
    config: AppConfig
    objectives: list[Callable[[np.ndarray], float]]

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        values = np.zeros((population.shape[0], len(self.objectives)), dtype=float)
        for idx, individual in enumerate(population):
            for obj_idx, obj_fn in enumerate(self.objectives):
                values[idx, obj_idx] = obj_fn(individual)
        return values
