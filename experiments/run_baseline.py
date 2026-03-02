from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner


def build_timestamped_out_dir(base_out: str | Path, engine: str) -> Path:
    base_path = Path(base_out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_engine = engine.strip().lower().replace(" ", "_")
    stamped_name = f"{base_path.name}_{safe_engine}_{timestamp}"
    return base_path.parent / stamped_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline fertilization optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to yaml config",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/baseline",
        help="Output artifact directory",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["random_search", "deap_nsga2"],
        help="Optimization engine override",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamp suffix for output directory",
    )
    # Add flags for overriding config parameters
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--prototype-flags", type=str, choices=["True", "False"], help="Use prototype crossover")
    parser.add_argument("--coupled-flags", type=str, choices=["True", "False"], help="Use coupled mutation")
    parser.add_argument("--dynamic-elite-flags", type=str, choices=["True", "False"], help="Use dynamic elite retention")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    # Override config with args if provided
    if args.seed is not None:
        cfg.seed = args.seed
    if args.prototype_flags is not None:
        cfg.use_prototype_crossover = (args.prototype_flags == "True")
    if args.coupled_flags is not None:
        cfg.use_coupled_mutation = (args.coupled_flags == "True")
    if args.dynamic_elite_flags is not None:
        cfg.use_dynamic_elite_retention = (args.dynamic_elite_flags == "True")

    project_root = Path(__file__).resolve().parent.parent
    registry = build_default_registry(str(project_root))
    objective_fns = registry.resolve(cfg.objectives)
    problem = FertilizationProblem(config=cfg, objectives=objective_fns)

    runner = BaselineRunner(cfg, problem)
    selected_engine = args.engine or cfg.optimizer_engine
    out_dir = Path(args.out)
    if not args.no_timestamp:
        out_dir = build_timestamped_out_dir(out_dir, selected_engine)

    artifacts = runner.run(out_dir, engine=selected_engine)

    print("完成基线运行，输出文件：")
    print(f"- {artifacts.init_population_path}")
    print(f"- {artifacts.final_population_path}")
    print(f"- {artifacts.final_objective_path}")
    print(f"- {artifacts.pareto_front_path}")


if __name__ == "__main__":
    main()
