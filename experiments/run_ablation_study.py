import argparse
import subprocess
import sys
import csv
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
import os
os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")

from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, nondominated_mask
from _experiment_logging import experiment_log_context, run_and_stream


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Optimized Ablation Study")
    parser.add_argument("--runs", type=int, default=5, help="Number of seeds (Default: 5)")
    parser.add_argument("--out", type=str, help="Output directory (default: artifacts/ablation_benchmark_<timestamp>)")
    args = parser.parse_args()

    if args.out:
        artifacts_root = Path(args.out)
        if not artifacts_root.is_absolute():
            artifacts_root = project_root / args.out
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_root = project_root / "artifacts" / f"ablation_benchmark_{timestamp}"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    with experiment_log_context(artifacts_root, "ablation_benchmark", configure_root_logger=True):
        configs = {
            "Full_Scheme2": [
                "--prototype-flags=False",
                "--coupled-flags=True",
                "--dynamic-elite-flags=True",
                "--crossover-method=pcx",
            ],
            "w/o_Coupled_Mutation": [
                "--prototype-flags=False",
                "--coupled-flags=False",
                "--dynamic-elite-flags=True",
                "--crossover-method=pcx",
            ],
            "w/o_Dynamic_Elite": [
                "--prototype-flags=False",
                "--coupled-flags=True",
                "--dynamic-elite-flags=False",
                "--crossover-method=pcx",
            ],
            "w/o_PCX_(use_SBX)": [
                "--prototype-flags=False",
                "--coupled-flags=True",
                "--dynamic-elite-flags=True",
                "--crossover-method=sbx",
            ],
            "Baseline_NSGA-II": [
                "--prototype-flags=False",
                "--coupled-flags=False",
                "--dynamic-elite-flags=False",
                "--crossover-method=sbx",
            ],
        }

        for name, flags in configs.items():
            for run_idx in range(args.runs):
                seed = 42 + run_idx
                out_dir = artifacts_root / f"{name}/run_{run_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)

                final_obj_path = out_dir / "final_objectives.csv"
                if not final_obj_path.exists():
                    logger.info(f"Running {name} [Seed {seed}]...")
                    cmd = [
                        sys.executable,
                        str(project_root / "experiments/run_baseline.py"),
                        "--config", str(project_root / "configs/default.yaml"),
                        "--out", str(out_dir),
                        "--seed", str(seed),
                        "--no-timestamp",
                    ] + flags

                    run_and_stream(cmd, env=dict(os.environ))

        logger.info("Evaluating metrics...")
        all_objectives = []

        for name in configs.keys():
            for run_idx in range(args.runs):
                obj_path = artifacts_root / f"{name}/run_{run_idx}/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)
                all_objectives.append(vals)

        union = np.vstack(all_objectives)
        ref_front = union[nondominated_mask(union)]
        nadir = np.max(ref_front, axis=0) + 1e-4

        eval_results = []
        summary_metrics = []

        for name in configs.keys():
            hvs = []
            igds = []
            for run_idx in range(args.runs):
                obj_path = artifacts_root / f"{name}/run_{run_idx}/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)

                hv = hypervolume_monte_carlo(vals, nadir)
                igd_val = igd(vals, ref_front)
                hvs.append(hv)
                igds.append(igd_val)

                eval_results.append(
                    {
                        "ablation_variant": name,
                        "run": run_idx,
                        "hv": hv,
                        "igd": igd_val,
                    }
                )

            summary_metrics.append(
                {
                    "ablation_variant": name,
                    "hv_mean": np.mean(hvs),
                    "hv_std": np.std(hvs),
                    "igd_mean": np.mean(igds),
                    "igd_std": np.std(igds),
                }
            )

        runs_csv = artifacts_root / "run_metrics.csv"
        with runs_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ablation_variant", "run", "hv", "igd"])
            writer.writeheader()
            writer.writerows(eval_results)

        summary_csv = artifacts_root / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ablation_variant", "hv_mean", "hv_std", "igd_mean", "igd_std"],
            )
            writer.writeheader()
            writer.writerows(summary_metrics)

        logger.info(f"Metrics saved to {artifacts_root}")

        logger.info("Plotting combined Pareto Fronts...")
        fig = plt.figure(figsize=(10, 8))
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F1C40F"]

        if ref_front.shape[1] >= 3:
            ax = fig.add_subplot(111, projection="3d")
            for idx, name in enumerate(configs.keys()):
                obj_path = artifacts_root / f"{name}/run_0/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], label=name, alpha=0.7, c=colors[idx])
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        else:
            for idx, name in enumerate(configs.keys()):
                obj_path = artifacts_root / f"{name}/run_0/final_objectives.csv"
                vals = np.loadtxt(obj_path, delimiter=",")
                plt.scatter(vals[:, 0], vals[:, 1], label=name, alpha=0.7, c=colors[idx])
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")

        plt.legend()
        plt.title("Ablation Study Pareto Front Comparison (Run 0)")
        plt.tight_layout()
        plt.savefig(artifacts_root / "pareto_front_ablation.png", dpi=150)
        plt.close(fig)

        print("\n--- Summary Metrics ---")
        for row in summary_metrics:
            print(
                f"{row['ablation_variant']:>25}: HV={row['hv_mean']:.4f}±{row['hv_std']:.4f}, "
                f"IGD={row['igd_mean']:.4f}±{row['igd_std']:.4f}"
            )

if __name__ == "__main__":
    main()
