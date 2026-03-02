from __future__ import annotations

from pathlib import Path

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner


def test_baseline_runner_outputs_artifacts(tmp_path: Path) -> None:
    cfg_path = Path("configs/default.yaml")
    cfg = load_config(cfg_path)
    registry = build_default_registry()
    fns = registry.resolve(cfg.objectives)
    problem = FertilizationProblem(config=cfg, objectives=fns)

    runner = BaselineRunner(cfg, problem)
    artifacts = runner.run(tmp_path / "baseline")

    assert artifacts.init_population_path.exists()
    assert artifacts.final_population_path.exists()
    assert artifacts.final_objective_path.exists()
    assert artifacts.pareto_front_path.exists()
