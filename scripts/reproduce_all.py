from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键复现实验主流程（批跑->评估->图表->总表）")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--out-root", type=str, default="artifacts/repro", help="复现实验根目录前缀")
    parser.add_argument("--python-exe", type=str, default=None, help="指定 Python 可执行文件")

    parser.add_argument("--engines", type=str, default="deap_nsga2")
    parser.add_argument("--seeds", type=str, default="42,43")
    parser.add_argument("--pop-sizes", type=str, default="40")
    parser.add_argument("--generations", type=str, default="12")
    parser.add_argument("--prototype-flags", type=str, default="true,false")
    parser.add_argument("--coupled-flags", type=str, default="true")
    parser.add_argument("--dynamic-elite-flags", type=str, default="true,false")
    parser.add_argument("--surrogate-flags", type=str, default="false")

    parser.add_argument("--hv-samples", type=int, default=8000)
    parser.add_argument("--hv-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], command_log: list[list[str]]) -> None:
    command_log.append(cmd)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def safe_command_output(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, env=env, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"<failed: {' '.join(cmd)} | {type(exc).__name__}: {exc}>"


def write_environment_lock(repro_root: Path, pyexe: str, env: dict[str, str]) -> Path:
    lock_path = repro_root / "environment.lock.txt"
    freeze_text = safe_command_output([pyexe, "-m", "pip", "freeze"], env=env)
    lock_path.write_text(freeze_text + "\n", encoding="utf-8")
    return lock_path


def write_run_commands_log(repro_root: Path, command_log: list[list[str]]) -> Path:
    log_path = repro_root / "run_commands.log"
    lines = ["# Reproduction command log"]
    for idx, cmd in enumerate(command_log, start=1):
        lines.append(f"[{idx}] {' '.join(cmd)}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def write_git_state(repro_root: Path, repo_root: Path) -> Path:
    state_path = repro_root / "git_state.txt"
    branch = safe_command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    commit = safe_command_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = safe_command_output(["git", "status", "--short"], cwd=repo_root)
    changed = safe_command_output(["git", "diff", "--name-only"], cwd=repo_root)

    content = [
        f"branch: {branch}",
        f"commit: {commit}",
        "status:",
        status if status else "(clean or unavailable)",
        "changed_files:",
        changed if changed else "(none)",
    ]
    state_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return state_path


def latest_dir(parent: Path, pattern: str) -> Path:
    dirs = sorted([p for p in parent.glob(pattern) if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"未找到目录模式: {parent / pattern}")
    return dirs[-1]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    repro_root = repo_root / f"{args.out_root}_{ts}"
    repro_root.mkdir(parents=True, exist_ok=True)

    pyexe = args.python_exe or os.environ.get("PYTHON_EXECUTABLE") or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    command_log: list[list[str]] = []

    batch_out_prefix = repro_root / "batch"
    batch_cmd = [
        pyexe,
        "experiments/batch_run.py",
        "--config",
        args.config,
        "--out",
        str(batch_out_prefix),
        "--engines",
        args.engines,
        "--seeds",
        args.seeds,
        "--pop-sizes",
        args.pop_sizes,
        "--generations",
        args.generations,
        "--prototype-flags",
        args.prototype_flags,
        "--coupled-flags",
        args.coupled_flags,
        "--dynamic-elite-flags",
        args.dynamic_elite_flags,
        "--surrogate-flags",
        args.surrogate_flags,
    ]
    run_command(batch_cmd, cwd=repo_root, env=env, command_log=command_log)

    batch_root = latest_dir(repro_root, "batch_*")

    eval_cmd = [
        pyexe,
        "experiments/evaluate_batch.py",
        "--batch-root",
        str(batch_root.relative_to(repo_root)),
        "--samples",
        str(args.hv_samples),
        "--seed",
        str(args.hv_seed),
    ]
    run_command(eval_cmd, cwd=repo_root, env=env, command_log=command_log)

    eval_root = latest_dir(batch_root, "evaluation_*")
    paper_out = batch_root / f"paper_{ts}"

    figure_cmd = [
        pyexe,
        "experiments/export_paper_figures.py",
        "--batch-root",
        str(batch_root.relative_to(repo_root)),
        "--evaluation-dir",
        str(eval_root.relative_to(repo_root)),
        "--out-dir",
        str(paper_out.relative_to(repo_root)),
        "--dpi",
        str(args.dpi),
    ]
    run_command(figure_cmd, cwd=repo_root, env=env, command_log=command_log)

    report_cmd = [
        pyexe,
        "experiments/build_result_report.py",
        "--batch-root",
        str(batch_root.relative_to(repo_root)),
        "--evaluation-dir",
        str(eval_root.relative_to(repo_root)),
        "--out-dir",
        str(paper_out.relative_to(repo_root)),
    ]
    run_command(report_cmd, cwd=repo_root, env=env, command_log=command_log)

    env_lock_path = write_environment_lock(repro_root, pyexe=pyexe, env=env)
    cmd_log_path = write_run_commands_log(repro_root, command_log=command_log)
    git_state_path = write_git_state(repro_root, repo_root=repo_root)

    manifest: dict[str, Any] = {
        "timestamp": ts,
        "repo_root": str(repo_root),
        "repro_root": str(repro_root),
        "batch_root": str(batch_root),
        "evaluation_root": str(eval_root),
        "paper_root": str(paper_out),
        "python_executable": pyexe,
        "commands": command_log,
        "lock_files": {
            "environment_lock": str(env_lock_path),
            "run_commands_log": str(cmd_log_path),
            "git_state": str(git_state_path),
        },
        "params": {
            "config": args.config,
            "engines": args.engines,
            "seeds": args.seeds,
            "pop_sizes": args.pop_sizes,
            "generations": args.generations,
            "prototype_flags": args.prototype_flags,
            "coupled_flags": args.coupled_flags,
            "dynamic_elite_flags": args.dynamic_elite_flags,
            "surrogate_flags": args.surrogate_flags,
            "hv_samples": args.hv_samples,
            "hv_seed": args.hv_seed,
            "dpi": args.dpi,
        },
    }

    manifest_path = repro_root / "repro_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("复现流程完成：")
    print(f"- 批次目录: {batch_root}")
    print(f"- 评估目录: {eval_root}")
    print(f"- 论文素材目录: {paper_out}")
    print(f"- 复现清单: {manifest_path}")


if __name__ == "__main__":
    main()
