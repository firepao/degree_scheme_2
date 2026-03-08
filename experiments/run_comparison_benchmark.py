from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner
from fertopt.baselines.external import (
    get_moead_runner,
    get_agemoea_runner,
    get_ctaea_runner,
    get_rvea_runner,
    get_smsemoa_runner,
    get_nsga3_runner
)
from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, nondominated_mask
from _experiment_logging import experiment_log_context

logger = logging.getLogger(__name__)

ALGO_MAP = {
    "scheme2": None, # Special case for our own runner
    "nsga3": get_nsga3_runner,
    "moead": get_moead_runner,
    "agemoea": get_agemoea_runner,
    "ctaea": get_ctaea_runner,
    "rvea": get_rvea_runner,
    "smsemoa": get_smsemoa_runner,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-seed Comparison Benchmark")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, help="Output directory (default: artifacts/comparison_benchmark_<timestamp>)")
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def resolve_output_dir(project_root: Path, output_arg: str | None) -> Path:
    if output_arg:
        out_dir = Path(output_arg)
        if not out_dir.is_absolute():
            out_dir = project_root / output_arg
        return out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "artifacts" / f"comparison_benchmark_{timestamp}"

def main():
    args = parse_args()
    out_dir = resolve_output_dir(project_root, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with experiment_log_context(out_dir, "comparison_benchmark", configure_root_logger=True):
        cfg_path = project_root / args.config
        base_cfg = load_config(str(cfg_path))
        registry = build_default_registry(str(project_root))
        objective_fns = registry.resolve(base_cfg.objectives)

        algorithms = list(ALGO_MAP.keys())

        all_objectives = []

        for algo in algorithms:
            for run_idx in range(args.runs):
                seed = 42 + run_idx
                run_dir = out_dir / algo / f"run_{run_idx}"
                if (run_dir / "final_objectives.csv").exists():
                    logger.info(f"Skipping {algo} run {run_idx}, already exists.")
                    continue

                cfg = load_config(str(cfg_path))
                cfg.seed = seed
                problem = FertilizationProblem(config=cfg, objectives=objective_fns)

                logger.info(f"Running {algo} [Seed {seed}]")
                if algo == "scheme2":
                    runner = BaselineRunner(cfg, problem)
                else:
                    runner = ALGO_MAP[algo](cfg, problem)

                runner.run(run_dir)

        logger.info("Evaluating metrics...")
        eval_results = []
        for algo in algorithms:
            for run_idx in range(args.runs):
                obj_path = out_dir / algo / f"run_{run_idx}" / "final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",", dtype=float)
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)
                all_objectives.append(vals)

        union = np.vstack(all_objectives)
        ref_front = union[nondominated_mask(union)]
        nadir = np.max(ref_front, axis=0) + 1e-4

        summary_metrics = []
        for algo in algorithms:
            hvs = []
            igds = []
            for run_idx in range(args.runs):
                obj_path = out_dir / algo / f"run_{run_idx}" / "final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",", dtype=float)
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)

                hv = hypervolume_monte_carlo(vals, nadir)
                igd_val = igd(vals, ref_front)
                hvs.append(hv)
                igds.append(igd_val)

                eval_results.append({
                    "algorithm": algo,
                    "run": run_idx,
                    "hv": hv,
                    "igd": igd_val,
                })

            summary_metrics.append({
                "algorithm": algo,
                "hv_mean": np.mean(hvs),
                "hv_std": np.std(hvs),
                "igd_mean": np.mean(igds),
                "igd_std": np.std(igds),
            })

        runs_csv = out_dir / "run_metrics.csv"
        with runs_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", "run", "hv", "igd"])
            writer.writeheader()
            writer.writerows(eval_results)

        summary_csv = out_dir / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["algorithm", "hv_mean", "hv_std", "igd_mean", "igd_std"],
            )
            writer.writeheader()
            writer.writerows(summary_metrics)

        logger.info(f"Summary metrics saved to {summary_csv}")

        logger.info("Plotting combined Pareto Fronts...")
        fig = plt.figure(figsize=(10, 8))
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F1C40F", "#34495E", "#E67E22"]

        if ref_front.shape[1] >= 3:
            ax = fig.add_subplot(111, projection="3d")
            for idx, algo in enumerate(algorithms):
                obj_path = out_dir / algo / "run_0" / "final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",", dtype=float)
                ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], label=algo, alpha=0.7, c=colors[idx])
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        else:
            for idx, algo in enumerate(algorithms):
                obj_path = out_dir / algo / "run_0" / "final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",", dtype=float)
                plt.scatter(vals[:, 0], vals[:, 1], label=algo, alpha=0.7, c=colors[idx])
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")

        plt.legend()
        plt.title("Pareto Front Comparison (Run 0)")
        plt.tight_layout()
        plt.savefig(out_dir / "pareto_front_comparison.png", dpi=150)
        plt.close(fig)

        print("\n--- Summary Metrics ---")
        for row in summary_metrics:
            print(
                f"{row['algorithm']:>10}: HV={row['hv_mean']:.4f}±{row['hv_std']:.4f}, "
                f"IGD={row['igd_mean']:.4f}±{row['igd_std']:.4f}"
            )

if __name__ == "__main__":
    main()
