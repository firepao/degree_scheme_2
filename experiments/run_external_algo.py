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
# Import the new external runner
from fertopt.baselines.external.nsga3 import NSGA3Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Run External Comparison Algorithm (NSGA-III)")
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

    # Initialize NSGA-III Runner
    runner = NSGA3Runner(cfg, problem)
    
    # Run
    out_dir = Path(args.out)
    runner.run(out_dir)

if __name__ == "__main__":
    main()
