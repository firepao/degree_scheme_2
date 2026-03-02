from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, nondominated_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate batch experiment outputs")
    parser.add_argument("--batch-root", type=str, default=None, help="Batch root directory containing manifest.csv")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.csv")
    parser.add_argument("--samples", type=int, default=20000, help="Monte Carlo samples for HV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for HV Monte Carlo")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_objectives(path_str: str) -> np.ndarray:
    vals = np.loadtxt(path_str, delimiter=",", dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


def main() -> None:
    args = parse_args()
    if (args.batch_root is None) == (args.manifest is None):
        raise ValueError("必须且只能提供 --batch-root 或 --manifest 之一")

    manifest_path = Path(args.manifest) if args.manifest else Path(args.batch_root) / "manifest.csv"
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest 为空，无法评估")

    objective_sets: list[np.ndarray] = []
    for rec in records:
        objective_sets.append(load_objectives(rec["final_objectives"]))

    union = np.vstack(objective_sets)
    ref_front = union[nondominated_mask(union)]
    ref_point = np.max(union, axis=0) + 1.0

    run_rows: list[dict[str, str]] = []
    for rec, obj in zip(records, objective_sets):
        hv = hypervolume_monte_carlo(obj, ref_point=ref_point, samples=args.samples, seed=args.seed)
        igd_value = igd(obj, ref_front)

        row = dict(rec)
        row["hv"] = f"{hv:.6f}"
        row["igd"] = f"{igd_value:.6f}"
        run_rows.append(row)

    group_keys = [
        "engine",
        "use_prototype_crossover",
        "use_coupled_mutation",
        "use_dynamic_elite_retention",
        "surrogate_enabled",
        "population_size",
        "num_generations",
    ]

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        gk = tuple(row.get(k, "") for k in group_keys)
        groups[gk].append(row)

    summary_rows: list[dict[str, str]] = []
    for gk, items in groups.items():
        hv_vals = np.array([float(it["hv"]) for it in items], dtype=float)
        igd_vals = np.array([float(it["igd"]) for it in items], dtype=float)
        time_vals = np.array([float(it.get("elapsed_seconds", "nan")) for it in items], dtype=float)

        row: dict[str, str] = {k: v for k, v in zip(group_keys, gk)}
        row["runs"] = str(len(items))
        row["hv_mean"] = f"{np.nanmean(hv_vals):.6f}"
        row["hv_std"] = f"{np.nanstd(hv_vals):.6f}"
        row["igd_mean"] = f"{np.nanmean(igd_vals):.6f}"
        row["igd_std"] = f"{np.nanstd(igd_vals):.6f}"
        row["elapsed_mean"] = f"{np.nanmean(time_vals):.6f}"
        row["elapsed_std"] = f"{np.nanstd(time_vals):.6f}"
        summary_rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = manifest_path.parent / f"evaluation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_out = out_dir / "run_metrics.csv"
    summary_out = out_dir / "summary_metrics.csv"

    with run_out.open("w", encoding="utf-8", newline="") as f:
        fnames = list(run_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        writer.writerows(run_rows)

    with summary_out.open("w", encoding="utf-8", newline="") as f:
        fnames = group_keys + [
            "runs",
            "hv_mean",
            "hv_std",
            "igd_mean",
            "igd_std",
            "elapsed_mean",
            "elapsed_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("评估完成：")
    print(f"- 运行级结果: {run_out}")
    print(f"- 分组汇总: {summary_out}")


if __name__ == "__main__":
    main()
