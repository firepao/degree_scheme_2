from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner


def parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("参数列表不能为空")
    return items


def parse_int_list(value: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"无法解析整型列表: {value}") from exc
    if not items:
        raise ValueError("参数列表不能为空")
    return items


def parse_bool_list(value: str) -> list[bool]:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    out: list[bool] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"无法解析布尔列表项: {item}")
        out.append(mapping[key])
    if not out:
        raise ValueError("布尔参数列表不能为空")
    return out


def build_batch_root(base_out: str | Path) -> Path:
    base_path = Path(base_out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path.parent / f"{base_path.name}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run fertilization experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument("--out", type=str, default="artifacts/batch", help="Batch output root directory")
    parser.add_argument(
        "--engines",
        type=str,
        default="random_search",
        help="Comma-separated engines, e.g. random_search,deap_nsga2",
    )
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    parser.add_argument("--pop-sizes", type=str, default=None, help="Comma-separated population sizes")
    parser.add_argument("--generations", type=str, default=None, help="Comma-separated generation counts")
    parser.add_argument("--prototype-flags", type=str, default=None, help="Comma-separated bools for prototype crossover")
    parser.add_argument("--coupled-flags", type=str, default=None, help="Comma-separated bools for coupled mutation")
    parser.add_argument("--dynamic-elite-flags", type=str, default=None, help="Comma-separated bools for dynamic elite retention")
    parser.add_argument("--surrogate-flags", type=str, default=None, help="Comma-separated bools for surrogate enable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    engines = parse_str_list(args.engines)
    seeds = parse_int_list(args.seeds) if args.seeds else [base_cfg.seed]
    pop_sizes = parse_int_list(args.pop_sizes) if args.pop_sizes else [base_cfg.population_size]
    generations = parse_int_list(args.generations) if args.generations else [base_cfg.num_generations]
    prototype_flags = parse_bool_list(args.prototype_flags) if args.prototype_flags else [base_cfg.use_prototype_crossover]
    coupled_flags = parse_bool_list(args.coupled_flags) if args.coupled_flags else [base_cfg.use_coupled_mutation]
    dynamic_flags = parse_bool_list(args.dynamic_elite_flags) if args.dynamic_elite_flags else [base_cfg.use_dynamic_elite_retention]
    surrogate_flags = parse_bool_list(args.surrogate_flags) if args.surrogate_flags else [base_cfg.surrogate.enabled]

    batch_root = build_batch_root(args.out)
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_root / "manifest.csv"

    project_root = Path(__file__).resolve().parent.parent
    registry = build_default_registry(str(project_root))
    run_records: list[dict[str, str]] = []

    for (
        engine,
        seed,
        pop_size,
        generation,
        prototype_flag,
        coupled_flag,
        dynamic_flag,
        surrogate_flag,
    ) in product(
        engines,
        seeds,
        pop_sizes,
        generations,
        prototype_flags,
        coupled_flags,
        dynamic_flags,
        surrogate_flags,
    ):
        cfg = replace(
            base_cfg,
            seed=seed,
            population_size=pop_size,
            num_generations=generation,
            use_prototype_crossover=prototype_flag,
            use_coupled_mutation=coupled_flag,
            use_dynamic_elite_retention=dynamic_flag,
            surrogate=replace(base_cfg.surrogate, enabled=surrogate_flag),
        )
        objective_fns = registry.resolve(cfg.objectives)
        problem = FertilizationProblem(config=cfg, objectives=objective_fns)
        runner = BaselineRunner(cfg, problem)

        run_name = (
            f"baseline_{engine}_seed{seed}_pop{pop_size}_gen{generation}"
            f"_proto{int(prototype_flag)}_coup{int(coupled_flag)}"
            f"_elite{int(dynamic_flag)}_surr{int(surrogate_flag)}"
        )
        run_dir = batch_root / run_name
        t0 = perf_counter()
        artifacts = runner.run(run_dir, engine=engine)
        elapsed = perf_counter() - t0

        run_records.append(
            {
                "engine": engine,
                "seed": str(seed),
                "population_size": str(pop_size),
                "num_generations": str(generation),
                "use_prototype_crossover": str(prototype_flag),
                "use_coupled_mutation": str(coupled_flag),
                "use_dynamic_elite_retention": str(dynamic_flag),
                "surrogate_enabled": str(surrogate_flag),
                "elapsed_seconds": f"{elapsed:.6f}",
                "run_dir": str(run_dir),
                "init_distribution": str(artifacts.init_population_path),
                "final_population": str(artifacts.final_population_path),
                "final_objectives": str(artifacts.final_objective_path),
                "pareto_front": str(artifacts.pareto_front_path),
            }
        )
        print(f"完成: {run_name}")

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "engine",
                "seed",
                "population_size",
                "num_generations",
                "use_prototype_crossover",
                "use_coupled_mutation",
                "use_dynamic_elite_retention",
                "surrogate_enabled",
                "elapsed_seconds",
                "run_dir",
                "init_distribution",
                "final_population",
                "final_objectives",
                "pareto_front",
            ],
        )
        writer.writeheader()
        writer.writerows(run_records)

    print("批量实验完成：")
    print(f"- 批次目录: {batch_root}")
    print(f"- 清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
