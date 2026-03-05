from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner


def test_deap_engine_outputs_artifacts(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, population_size=40, num_generations=8)

    registry = build_default_registry()
    fns = registry.resolve(cfg.objectives)
    problem = FertilizationProblem(config=cfg, objectives=fns)
    runner = BaselineRunner(cfg, problem)

    artifacts = runner.run(tmp_path / "baseline_deap", engine="deap_nsga2")

    assert artifacts.init_population_path.exists()
    assert artifacts.final_population_path.exists()
    assert artifacts.final_objective_path.exists()
    assert artifacts.pareto_front_path.exists()
