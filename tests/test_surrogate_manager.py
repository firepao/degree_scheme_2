from __future__ import annotations

import numpy as np
import pytest

from fertopt.models.surrogate import SurrogateManager, SurrogateParams


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm 未安装，跳过代理模型测试",
)
def test_surrogate_manager_initialize_predict_and_active_update() -> None:
    rng = np.random.default_rng(99)
    x_init = rng.uniform(0.0, 1.0, size=(30, 6))

    def true_eval(x: np.ndarray) -> np.ndarray:
        y1 = np.sum(x, axis=1)
        y2 = np.sum(x ** 2, axis=1)
        y3 = np.sum(np.abs(x - 0.5), axis=1)
        return np.stack([y1, y2, y3], axis=1)

    y_init = true_eval(x_init)

    mgr = SurrogateManager(
        objective_names=["yield", "cost", "nitrogen_loss"],
        params=SurrogateParams(
            enabled=True,
            update_interval_g=2,
            query_batch_size=5,
            target_objectives=["yield", "nitrogen_loss"],
            model_num_estimators=50,
            model_learning_rate=0.1,
            seed=42,
        ),
    )

    mgr.initialize(x_init, y_init)
    before = mgr.training_size()

    x_query = rng.uniform(0.0, 1.0, size=(10, 6))
    y_pred = mgr.predict_objectives(x_query, true_eval)

    assert y_pred.shape == (10, 3)

    updated = mgr.active_update(
        generation_index=2,
        x_candidates=x_query,
        true_eval_fn=true_eval,
    )
    after = mgr.training_size()

    assert updated is True
    assert after > before
