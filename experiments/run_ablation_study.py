import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
import os
os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")

from fertopt.core.objectives import build_default_registry
from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, spacing, nondominated_mask

def run_experiment(name, flags, out_dir, seed):
    """Run a single experiment configuration."""
    cmd = [
        sys.executable,
        str(project_root / "experiments/run_baseline.py"),
        "--config", str(project_root / "configs/default.yaml"),
        "--out", str(out_dir),
        "--seed", str(seed),
        "--no-timestamp" # We manage directories manually
    ] + flags
    
    print(f"Running {name} (Seed {seed})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {name}: {result.stderr}")
        return None
    
    return out_dir

def calculate_metrics(out_dir, ref_point):
    """Calculate metrics for a run."""
    try:
        obj_path = out_dir / "final_objectives.csv"
        if not obj_path.exists():
            return None
            
        objs = np.loadtxt(obj_path, delimiter=",")
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
            
        # Hypervolume Calculation
        # The objectives are:
        # 1. Yield (Minimizing negative yield, i.e., -80) -> Better is more negative.
        # 2. Cost (Minimizing positive cost, i.e., 5000) -> Better is smaller.
        # 3. N_Loss (Minimizing positive loss, i.e., 20) -> Better is smaller.
        
        # fertopt.evaluation.metrics.hypervolume_monte_carlo computes volume dominated by minimization front
        # relative to a reference point that is LARGER (worse) than all points.
        
        # So we can use `objs` directly as minimization objectives.
        # Reference Point: Should be worse (larger) than all points.
        # Yield range ~ [-100, -20]. Worse is 0.
        # Cost range ~ [2000, 20000]. Worse is 30000.
        # Loss range ~ [5, 100]. Worse is 200.
        
        ref_point_min = np.array([0.0, 30000.0, 200.0])
        hv = hypervolume_monte_carlo(objs, ref_point_min)
        return hv
    except Exception as e:
        print(f"Metric calc error: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 789, 999], help="Random seeds (Default: 5 seeds)")
    parser.add_argument("--resume", type=str, help="Resume from specific timestamp suffix (e.g. 20260303_181357)")
    args = parser.parse_args()

    if args.resume:
        timestamp = args.resume
        artifacts_root = project_root / f"artifacts/ablation_study_{timestamp}"
    else:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # artifacts_root = project_root / f"artifacts/ablation_study_{timestamp}"
        artifacts_root = project_root / "artifacts/final_ultimate_benchmark"
    
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # 1. Define Configurations for Ablation Study
    # Baseline: NSGA-II (No custom modules)
    # Modules:
    #   - SBC: Synergistic Balance Crossover
    #   - PCX: Parent-Centric Crossover (新添加)
    #   - Coupled: Coupled Mutation
    #   - Dynamic: Dynamic Elite Retention
    # Full: All modules ON
    
    # Define methods to compare
    configs = {
        "Baseline_NSGA-II": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False",
            "--crossover-method=sbx" 
        ],
        "Ultimate_Model (Dynamic+PCX+Coupled)": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True",
            "--crossover-method=pcx"
        ]
    }

    # Run Experiments
    results = []
    
    # Reference Point for Hypervolume (Minimization)
    ref_point_min = np.array([0.0, 30000.0, 200.0])

    for name, flags in configs.items():
        for seed in args.seeds:
            out_dir = artifacts_root / f"{name}_seed{seed}"
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
                
            # Check if execution is needed
            final_obj_path = out_dir / "final_objectives.csv"
            if not final_obj_path.exists():
                print(f"Running {name} (Seed {seed})...")
                
                if name == "External_NSGA-III":
                    # Run External Algorithm Script
                    cmd = [
                        sys.executable,
                        str(project_root / "experiments/run_external_algo.py"),
                        "--config", str(project_root / "configs/default.yaml"),
                        "--out", str(out_dir),
                        "--seed", str(seed)
                    ]
                else:
                    # Run Standard Baseline Script
                    cmd = [
                        sys.executable,
                        str(project_root / "experiments/run_baseline.py"),
                        "--config", str(project_root / "configs/default.yaml"),
                        "--out", str(out_dir),
                        "--seed", str(seed),
                        "--no-timestamp"
                    ] + flags
                
                subprocess.run(cmd, check=True)
            
            # Analyze Result
            objs = np.loadtxt(final_obj_path, delimiter=",")
            if objs.ndim == 1: objs = objs.reshape(1, -1)
            
            results.append({
                "Method": name,
                "Seed": seed,
                "Objectives": objs,
                "Yield": -objs[:, 0], # Back to positive
                "Cost": objs[:, 1],
                "N_Loss": objs[:, 2]
            })

    # 2. Analyze & Visualize
    if not results:
        print("No results collected.")
        return

    # Calculate Global Pareto Front for IGD
    all_objs = np.vstack([r["Objectives"] for r in results])
    global_mask = nondominated_mask(all_objs)
    global_pf = all_objs[global_mask]
    
    # Calculate Metrics
    for r in results:
        r["Hypervolume"] = hypervolume_monte_carlo(r["Objectives"], ref_point_min)
        r["IGD"] = igd(r["Objectives"], global_pf)
        r["Spacing"] = spacing(r["Objectives"])
        print(f"{r['Method']} (Seed {r['Seed']}) HV: {r['Hypervolume']:.2e}, IGD: {r['IGD']:.2f}, Spacing: {r['Spacing']:.2f}")

    df = pd.DataFrame(results)
    
    # Boxplot for HV
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Method", y="Hypervolume", palette="Set2")
    plt.title("Ablation Study: Hypervolume (Higher is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(artifacts_root / "ablation_hv_boxplot.png")
    
    # Boxplot for IGD
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Method", y="IGD", palette="Set2")
    plt.title("Ablation Study: IGD (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(artifacts_root / "ablation_igd_boxplot.png")

    print(f"Saved boxplots to {artifacts_root}")

    # Pareto Front Visualization (Merge all seeds)
    plt.figure(figsize=(10, 8))
    
    # 2D Plot: Cost vs Yield (Most important trade-off)
    for name in configs.keys():
        subset = df[df["Method"] == name]
        all_yields = np.concatenate(subset["Yield"].values)
        all_costs = np.concatenate(subset["Cost"].values)
        
        plt.scatter(all_costs, all_yields, label=name, alpha=0.6, s=30)
        
    plt.xlabel("Cost (Lower is Better)")
    plt.ylabel("Yield (Higher is Better)")
    plt.title("Pareto Front Distribution: Cost vs Yield")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(artifacts_root / "pareto_comparison_2d.png")
    print("Saved Pareto 2D comparison to pareto_comparison_2d.png")

    print("\nFinal Comparison Completed!")

if __name__ == "__main__":
    main()
