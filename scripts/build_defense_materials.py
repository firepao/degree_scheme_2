from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成答辩材料目录（图表/总表/摘要/版本信息）")
    parser.add_argument("--batch-root", type=str, required=True, help="批次目录（用于追溯）")
    parser.add_argument("--paper-dir", type=str, default=None, help="论文素材目录（paper_*）；不传则自动选最新")
    parser.add_argument("--out-root", type=str, default="deliverables", help="答辩材料输出根目录")
    return parser.parse_args()


def latest_paper_dir(batch_root: Path) -> Path:
    candidates = sorted([p for p in batch_root.glob("paper_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"未在 {batch_root} 下找到 paper_* 目录")
    return candidates[-1]


def read_table_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_pm(value: str) -> float:
    text = value.strip()
    if "±" in text:
        text = text.split("±")[0].strip()
    return float(text)


def safe_git(cmd: list[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "N/A"


def build_key_findings(table_rows: list[dict[str, str]], out_path: Path) -> None:
    if not table_rows:
        out_path.write_text("# 关键结论\n\n暂无可用结果。\n", encoding="utf-8")
        return

    best_hv = max(table_rows, key=lambda r: parse_pm(r.get("HV", "0")))
    best_igd = min(table_rows, key=lambda r: parse_pm(r.get("IGD", "999999")))
    best_runtime = min(table_rows, key=lambda r: parse_pm(r.get("Runtime(s)", "999999")))

    lines = [
        "# 关键结论",
        "",
        "## 1) 综合性能（HV 最大）",
        f"- 场景: {best_hv.get('scenario', 'N/A')}",
        f"- HV: {best_hv.get('HV', 'N/A')}",
        f"- IGD: {best_hv.get('IGD', 'N/A')}",
        f"- Runtime(s): {best_hv.get('Runtime(s)', 'N/A')}",
        "",
        "## 2) 收敛质量（IGD 最小）",
        f"- 场景: {best_igd.get('scenario', 'N/A')}",
        f"- HV: {best_igd.get('HV', 'N/A')}",
        f"- IGD: {best_igd.get('IGD', 'N/A')}",
        f"- Runtime(s): {best_igd.get('Runtime(s)', 'N/A')}",
        "",
        "## 3) 运行效率（Runtime 最短）",
        f"- 场景: {best_runtime.get('scenario', 'N/A')}",
        f"- HV: {best_runtime.get('HV', 'N/A')}",
        f"- IGD: {best_runtime.get('IGD', 'N/A')}",
        f"- Runtime(s): {best_runtime.get('Runtime(s)', 'N/A')}",
        "",
        "## 4) 口播建议",
        "- 先讲 Main 与消融对比，再讲效率与稳定性。",
        "- 强调所有图表均可追溯到同一批次输出目录。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_method_overview(out_path: Path) -> None:
    lines = [
        "# 方法摘要（答辩版）",
        "",
        "1. 初始化：基于 Beta 分布的连续型偏向初始化，覆盖低/中/高施肥区域。",
        "2. 交叉：原型引导交叉，将子代向高质量施肥原型牵引。",
        "3. 变异：协同/拮抗耦合变异，将 N-P-K 先验编码到扰动协方差。",
        "4. 选择：动态精英保留，融合非支配等级与局部稀疏度。",
        "5. 评估：批量实验后统一计算 HV、IGD 与运行时间，并自动导出论文图表与总表。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_version_info(out_path: Path, repo_root: Path, batch_root: Path, paper_dir: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    branch = safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    commit = safe_git(["git", "rev-parse", "HEAD"], repo_root)
    status = safe_git(["git", "status", "--short"], repo_root)

    lines = [
        f"generated_at: {now}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"repo_root: {repo_root}",
        f"batch_root: {batch_root}",
        f"paper_dir: {paper_dir}",
        f"git_branch: {branch}",
        f"git_commit: {commit}",
        "git_status:",
        status if status else "(clean or unavailable)",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    batch_root = (repo_root / args.batch_root).resolve()
    paper_dir = (repo_root / args.paper_dir).resolve() if args.paper_dir else latest_paper_dir(batch_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / args.out_root / f"defense_{ts}").resolve()
    slides_dir = out_dir / "slides_assets"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for fig in sorted(paper_dir.glob("fig_*.png")):
        shutil.copy2(fig, slides_dir / fig.name)

    for table_file in sorted(paper_dir.glob("table_main_ablation.*")):
        shutil.copy2(table_file, tables_dir / table_file.name)

    table_csv = tables_dir / "table_main_ablation.csv"
    rows = read_table_csv(table_csv) if table_csv.exists() else []

    key_findings_path = out_dir / "key_findings.md"
    method_overview_path = out_dir / "method_overview.md"
    version_info_path = out_dir / "version_info.txt"

    build_key_findings(rows, key_findings_path)
    build_method_overview(method_overview_path)
    build_version_info(version_info_path, repo_root=repo_root, batch_root=batch_root, paper_dir=paper_dir)

    summary = {
        "batch_root": str(batch_root),
        "paper_dir": str(paper_dir),
        "defense_dir": str(out_dir),
        "slides_assets": [p.name for p in sorted(slides_dir.glob("*.png"))],
        "tables": [p.name for p in sorted(tables_dir.glob("*"))],
        "key_findings": str(key_findings_path),
        "method_overview": str(method_overview_path),
        "version_info": str(version_info_path),
    }

    manifest_path = out_dir / "defense_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("答辩材料目录生成完成：")
    print(f"- 输出目录: {out_dir}")
    print(f"- 清单: {manifest_path}")


if __name__ == "__main__":
    main()
