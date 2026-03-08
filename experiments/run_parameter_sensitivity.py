import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.core.runner import BaselineRunner
from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, nondominated_mask
from _experiment_logging import experiment_log_context

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Parameter Sensitivity Benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Number of seeds (Default: 5)")
    parser.add_argument("--out", type=str, help="Output directory (default: artifacts/parameter_sensitivity_<timestamp>)")
    args = parser.parse_args()

    if args.out:
        artifacts_root = Path(args.out)
        if not artifacts_root.is_absolute():
            artifacts_root = project_root / args.out
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_root = project_root / "artifacts" / f"parameter_sensitivity_{timestamp}"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    with experiment_log_context(artifacts_root, "parameter_sensitivity", configure_root_logger=True):
        base_cfg = load_config(str(project_root / "configs/default.yaml"))

        param_grid = {
            "population_size": [40, 80, 120, 160],
            "beta_strength_k": [2.0, 4.0, 8.0, 12.0],
            "crossover.gamma0": [0.01, 0.05, 0.1, 0.2],
            "selection.alpha0": [0.0, 0.2, 0.5, 0.8],
        }

        experiments = []
        for param_name, values in param_grid.items():
            for val in values:
                exp_name = f"{param_name}_{val}"
                experiments.append((param_name, val, exp_name))

        registry = build_default_registry(str(project_root))
        objective_fns = registry.resolve(base_cfg.objectives)

        for param_name, val, exp_name in experiments:
            for run_idx in range(args.runs):
                seed = 42 + run_idx
                out_dir = artifacts_root / f"{exp_name}/run_{run_idx}"

                final_obj_path = out_dir / "final_objectives.csv"
                if not final_obj_path.exists():
                    print(f"Running {exp_name} [Seed {seed}]...")
                    out_dir.mkdir(parents=True, exist_ok=True)

                    cfg = copy.deepcopy(base_cfg)
                    cfg.seed = seed

                    if "." in param_name:
                        parent, child = param_name.split(".")
                        setattr(getattr(cfg, parent), child, val)
                    else:
                        setattr(cfg, param_name, val)

                    cfg.use_prototype_crossover = True
                    cfg.use_coupled_mutation = True
                    cfg.use_dynamic_elite_retention = True

                    problem = FertilizationProblem(config=cfg, objectives=objective_fns)
                    runner = BaselineRunner(cfg, problem)
                    runner.run(out_dir)

        print("Evaluating metrics...")
        all_objectives = []
        for param_name, val, exp_name in experiments:
            for run_idx in range(args.runs):
                obj_path = artifacts_root / f"{exp_name}/run_{run_idx}/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)
                all_objectives.append(vals)

        union = np.vstack(all_objectives)
        ref_front = union[nondominated_mask(union)]
        nadir = np.max(ref_front, axis=0) + 1e-4

        eval_results = []
        summary_metrics = []

        for param_name, val, exp_name in experiments:
            hvs = []
            igds = []
            for run_idx in range(args.runs):
                obj_path = artifacts_root / f"{exp_name}/run_{run_idx}/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)

                hv = hypervolume_monte_carlo(vals, nadir)
                igd_val = igd(vals, ref_front)
                hvs.append(hv)
                igds.append(igd_val)

                eval_results.append(
                    {
                        "parameter": param_name,
                        "value": val,
                        "run": run_idx,
                        "hv": hv,
                        "igd": igd_val,
                    }
                )

            summary_metrics.append(
                {
                    "parameter": param_name,
                    "value": val,
                    "hv_mean": np.mean(hvs),
                    "hv_std": np.std(hvs),
                    "igd_mean": np.mean(igds),
                    "igd_std": np.std(igds),
                }
            )

        runs_csv = artifacts_root / "run_metrics.csv"
        with runs_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parameter", "value", "run", "hv", "igd"])
            writer.writeheader()
            writer.writerows(eval_results)

        summary_csv = artifacts_root / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["parameter", "value", "hv_mean", "hv_std", "igd_mean", "igd_std"],
            )
            writer.writeheader()
            writer.writerows(summary_metrics)

        print(f"Metrics saved to {artifacts_root}")

        df_summary = pd.DataFrame(summary_metrics)

        for param_name in param_grid.keys():
            subset = df_summary[df_summary["parameter"] == param_name].sort_values("value")

            fig, ax1 = plt.subplots(figsize=(8, 6))

            color1 = "tab:red"
            ax1.set_xlabel(param_name)
            ax1.set_ylabel("Hypervolume (HV)", color=color1)
            ax1.errorbar(
                subset["value"],
                subset["hv_mean"],
                yerr=subset["hv_std"],
                fmt="-o",
                color=color1,
                capsize=5,
                label="HV",
            )
            ax1.tick_params(axis="y", labelcolor=color1)

            ax2 = ax1.twinx()
            color2 = "tab:blue"
            ax2.set_ylabel("IGD", color=color2)
            ax2.errorbar(
                subset["value"],
                subset["igd_mean"],
                yerr=subset["igd_std"],
                fmt="-s",
                color=color2,
                capsize=5,
                label="IGD",
            )
            ax2.tick_params(axis="y", labelcolor=color2)

            fig.tight_layout()
            plt.title(f"Parameter Sensitivity: {param_name}")

            clean_name = param_name.replace(".", "_")
            plt.savefig(artifacts_root / f"sensitivity_{clean_name}.png", dpi=150)
            plt.close(fig)

        print("All trend plots saved successfully!")

if __name__ == "__main__":
    main()
