from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class SurrogateParams:
    enabled: bool
    update_interval_g: int
    query_batch_size: int
    target_objectives: list[str]
    model_num_estimators: int
    model_learning_rate: float
    seed: int


class SurrogateManager:
    def __init__(self, objective_names: list[str], params: SurrogateParams) -> None:
        self.objective_names = objective_names
        self.params = params
        self._target_indices = [
            idx for idx, name in enumerate(objective_names) if name in set(params.target_objectives)
        ]

        self._models: dict[int, object] = {}
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    @property
    def is_enabled(self) -> bool:
        return self.params.enabled

    def initialize(self, x: np.ndarray, y_true: np.ndarray) -> None:
        if not self.is_enabled:
            return
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_y = np.asarray(y_true, dtype=float).copy()
        self._fit_models()

    def predict_objectives(self, x: np.ndarray, true_eval_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.is_enabled or self._train_x is None or self._train_y is None or not self._models:
            return true_eval_fn(x)

        y_pred = np.zeros((x.shape[0], len(self.objective_names)), dtype=float)

        fallback_indices = [idx for idx in range(len(self.objective_names)) if idx not in self._target_indices]
        if fallback_indices:
            y_true_part = true_eval_fn(x)
            y_pred[:, fallback_indices] = y_true_part[:, fallback_indices]

        for idx in self._target_indices:
            model = self._models.get(idx)
            if model is None:
                y_true_part = true_eval_fn(x)
                y_pred[:, idx] = y_true_part[:, idx]
            else:
                y_pred[:, idx] = model.predict(x)

        return y_pred

    def active_update(
        self,
        generation_index: int,
        x_candidates: np.ndarray,
        true_eval_fn: Callable[[np.ndarray], np.ndarray],
    ) -> bool:
        if not self.is_enabled:
            return False
        if generation_index <= 0:
            return False
        if generation_index % max(1, self.params.update_interval_g) != 0:
            return False
        if self._train_x is None or self._train_y is None:
            return False

        x_candidates = np.asarray(x_candidates, dtype=float)
        if x_candidates.shape[0] == 0:
            return False

        pick = min(self.params.query_batch_size, x_candidates.shape[0])
        selected_idx = self._select_query_indices(x_candidates, pick)
        x_query = x_candidates[selected_idx]
        y_query = true_eval_fn(x_query)

        self._train_x = np.vstack([self._train_x, x_query])
        self._train_y = np.vstack([self._train_y, y_query])
        self._fit_models()
        return True

    def training_size(self) -> int:
        return 0 if self._train_x is None else int(self._train_x.shape[0])

    def _fit_models(self) -> None:
        if self._train_x is None or self._train_y is None:
            return
        if not self._target_indices:
            return

        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "surrogate.enabled=True 但未安装 lightgbm，请先安装 lightgbm。"
            ) from exc

        self._models = {}
        for idx in self._target_indices:
            model = LGBMRegressor(
                n_estimators=self.params.model_num_estimators,
                learning_rate=self.params.model_learning_rate,
                random_state=self.params.seed + idx,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(self._train_x, self._train_y[:, idx])
            self._models[idx] = model

    def _select_query_indices(self, x_candidates: np.ndarray, pick: int) -> np.ndarray:
        if self._train_x is None or self._train_x.shape[0] == 0:
            return np.arange(pick)

        dist = np.linalg.norm(
            x_candidates[:, None, :] - self._train_x[None, :, :],
            axis=2,
        )
        nearest = np.min(dist, axis=1)
        return np.argsort(-nearest)[:pick]
