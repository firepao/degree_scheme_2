from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文结果总表（CSV + Markdown）")
    parser.add_argument("--batch-root", type=str, required=True, help="批次目录（包含 evaluation_*）")
    parser.add_argument("--evaluation-dir", type=str, default=None, help="指定评估目录；不传则自动选最新")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录；不传则自动创建 paper_<ts>")
    return parser.parse_args()


def latest_evaluation_dir(batch_root: Path) -> Path:
    eval_dirs = sorted([p for p in batch_root.glob("evaluation_*") if p.is_dir()])
    if not eval_dirs:
        raise FileNotFoundError(f"未在 {batch_root} 找到 evaluation_* 目录")
    return eval_dirs[-1]


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scenario_name(row: dict[str, str]) -> str:
    proto = row.get("use_prototype_crossover", "")
    coupled = row.get("use_coupled_mutation", "")
    elite = row.get("use_dynamic_elite_retention", "")
    surr = row.get("surrogate_enabled", "")

    if proto == "True" and coupled == "True" and elite == "True" and surr == "True":
        return "Main (All On)"
    if proto == "True" and coupled == "True" and elite == "True" and surr == "False":
        return "Main (No Surrogate)"
    if proto == "False" and coupled == "True" and elite == "True":
        return "Ablation - w/o Prototype"
    if proto == "True" and coupled == "False" and elite == "True":
        return "Ablation - w/o CoupledMutation"
    if proto == "True" and coupled == "True" and elite == "False":
        return "Ablation - w/o DynamicElite"
    if surr == "False":
        return "Ablation - w/o Surrogate"
    return (
        f"Custom(P={proto},C={coupled},E={elite},S={surr})"
    )


def format_pm(mean_str: str, std_str: str, digits: int = 4) -> str:
    mean = float(mean_str)
    std = float(std_str)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    batch_root = Path(args.batch_root)
    eval_dir = Path(args.evaluation_dir) if args.evaluation_dir else latest_evaluation_dir(batch_root)

    summary_csv = eval_dir / "summary_metrics.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"未找到 {summary_csv}")

    rows = read_summary_rows(summary_csv)

    report_rows: list[dict[str, str]] = []
    for row in rows:
        report_rows.append(
            {
                "scenario": scenario_name(row),
                "engine": row.get("engine", ""),
                "population_size": row.get("population_size", ""),
                "num_generations": row.get("num_generations", ""),
                "runs": row.get("runs", ""),
                "HV": format_pm(row["hv_mean"], row["hv_std"], digits=3),
                "IGD": format_pm(row["igd_mean"], row["igd_std"], digits=3),
                "Runtime(s)": format_pm(row["elapsed_mean"], row["elapsed_std"], digits=3),
            }
        )

    def sort_key(item: dict[str, str]) -> tuple[int, str]:
        scenario = item["scenario"]
        if scenario.startswith("Main"):
            return (0, scenario)
        if scenario.startswith("Ablation"):
            return (1, scenario)
        return (2, scenario)

    report_rows.sort(key=sort_key)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = batch_root / f"paper_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "table_main_ablation.csv"
    md_out = out_dir / "table_main_ablation.md"

    fieldnames = [
        "scenario",
        "engine",
        "population_size",
        "num_generations",
        "runs",
        "HV",
        "IGD",
        "Runtime(s)",
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    with md_out.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in report_rows:
            f.write("| " + " | ".join(row[k] for k in fieldnames) + " |\n")

    print("结果总表生成完成：")
    print(f"- CSV: {csv_out}")
    print(f"- Markdown: {md_out}")


if __name__ == "__main__":
    main()
