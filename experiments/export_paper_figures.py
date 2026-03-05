from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出论文图表（HV/IGD/耗时/帕累托叠加）")
    parser.add_argument("--batch-root", type=str, required=True, help="批次目录（包含 evaluation_*）")
    parser.add_argument("--evaluation-dir", type=str, default=None, help="指定评估目录；不传则自动选择最新 evaluation_*")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录；不传则自动创建 paper_<ts>")
    parser.add_argument("--dpi", type=int, default=300, help="图片分辨率")
    parser.add_argument(
        "--group-by",
        type=str,
        default="engine,use_prototype_crossover,use_coupled_mutation,use_dynamic_elite_retention,surrogate_enabled",
        help="分组字段，逗号分隔",
    )
    return parser.parse_args()


def latest_evaluation_dir(batch_root: Path) -> Path:
    eval_dirs = sorted([p for p in batch_root.glob("evaluation_*") if p.is_dir()])
    if not eval_dirs:
        raise FileNotFoundError(f"未在 {batch_root} 找到 evaluation_* 目录")
    return eval_dirs[-1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_group_label(row: dict[str, str], keys: list[str]) -> str:
    parts: list[str] = []
    for k in keys:
        v = row.get(k, "")
        if k == "engine":
            parts.append(v)
        elif k == "use_prototype_crossover":
            parts.append(f"Proto={v}")
        elif k == "use_coupled_mutation":
            parts.append(f"Coupled={v}")
        elif k == "use_dynamic_elite_retention":
            parts.append(f"Elite={v}")
        elif k == "surrogate_enabled":
            parts.append(f"Surr={v}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def save_boxplot(
    metric_name: str,
    run_rows: list[dict[str, str]],
    group_keys: list[str],
    out_path: Path,
    ylabel: str,
) -> None:
    grouped: dict[str, list[float]] = {}
    for row in run_rows:
        label = build_group_label(row, group_keys)
        grouped.setdefault(label, []).append(float(row[metric_name]))

    labels = list(grouped.keys())
    data = [grouped[label] for label in labels]

    plt.figure(figsize=(max(10, len(labels) * 1.6), 5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_runtime_bar(summary_rows: list[dict[str, str]], group_keys: list[str], out_path: Path) -> None:
    labels = [build_group_label(row, group_keys) for row in summary_rows]
    means = [float(row["elapsed_mean"]) for row in summary_rows]
    stds = [float(row["elapsed_std"]) for row in summary_rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 1.6), 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("平均运行时间（秒）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_pareto_overlay(run_rows: list[dict[str, str]], group_keys: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for row in run_rows:
        label = build_group_label(row, group_keys)
        obj_path = Path(row["final_objectives"])
        obj = np.loadtxt(obj_path, delimiter=",", dtype=float)
        if obj.ndim == 1:
            obj = obj.reshape(1, -1)
        plt.scatter(obj[:, 0], obj[:, 1], s=14, alpha=0.5, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=8)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Overlay")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    batch_root = Path(args.batch_root)
    eval_dir = Path(args.evaluation_dir) if args.evaluation_dir else latest_evaluation_dir(batch_root)

    run_csv = eval_dir / "run_metrics.csv"
    summary_csv = eval_dir / "summary_metrics.csv"
    if not run_csv.exists() or not summary_csv.exists():
        raise FileNotFoundError("评估目录缺少 run_metrics.csv 或 summary_metrics.csv")

    run_rows = read_csv_rows(run_csv)
    summary_rows = read_csv_rows(summary_csv)
    group_keys = [k.strip() for k in args.group_by.split(",") if k.strip()]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = batch_root / f"paper_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hv_path = out_dir / "fig_hv_boxplot.png"
    igd_path = out_dir / "fig_igd_boxplot.png"
    runtime_path = out_dir / "fig_runtime_bar.png"
    pareto_path = out_dir / "fig_pareto_overlay.png"

    save_boxplot("hv", run_rows, group_keys, hv_path, "HV")
    save_boxplot("igd", run_rows, group_keys, igd_path, "IGD")
    save_runtime_bar(summary_rows, group_keys, runtime_path)
    save_pareto_overlay(run_rows, group_keys, pareto_path)

    manifest = {
        "batch_root": str(batch_root),
        "evaluation_dir": str(eval_dir),
        "run_metrics": str(run_csv),
        "summary_metrics": str(summary_csv),
        "group_by": group_keys,
        "outputs": {
            "fig_hv_boxplot": str(hv_path),
            "fig_igd_boxplot": str(igd_path),
            "fig_runtime_bar": str(runtime_path),
            "fig_pareto_overlay": str(pareto_path),
        },
    }

    manifest_path = out_dir / "paper_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("论文图表导出完成：")
    print(f"- 输出目录: {out_dir}")
    print(f"- 清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
