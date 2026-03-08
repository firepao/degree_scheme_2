from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to sys.path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from fertopt.core.config import load_config
from fertopt.core.objectives import build_default_registry
from fertopt.core.problem import FertilizationProblem
from fertopt.baselines.external import (
    get_moead_runner,
    get_agemoea_runner,
    get_ctaea_runner,
    get_rvea_runner,
    get_smsemoa_runner,
    get_nsga3_runner
)

ALGO_MAP = {
    "nsga3": get_nsga3_runner,
    "moead": get_moead_runner,
    "agemoea": get_agemoea_runner,
    "ctaea": get_ctaea_runner,
    "rvea": get_rvea_runner,
    "smsemoa": get_smsemoa_runner,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run External Comparison Algorithm")
    parser.add_argument("--algo", type=str, required=True, choices=ALGO_MAP.keys(), help="Algorithm to run")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    cfg_path = project_root / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
        
    cfg = load_config(str(cfg_path))
    cfg.seed = args.seed
    
    # Setup problem
    registry = build_default_registry(str(project_root))
    objective_fns = registry.resolve(cfg.objectives)
    problem = FertilizationProblem(config=cfg, objectives=objective_fns)

    # Initialize Runner
    runner_factory = ALGO_MAP[args.algo.lower()]
    runner = runner_factory(cfg, problem)
    
    # Run
    out_dir = Path(args.out)
    runner.run(out_dir)

if __name__ == "__main__":
    main()
