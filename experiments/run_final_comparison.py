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
sys.path.append(str(project_root))

from fertopt.core.objectives import build_default_registry
from fertopt.evaluation.metrics import hypervolume_monte_carlo

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
    parser = argparse.ArgumentParser(description="Run Final Comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = project_root / f"artifacts/final_comparison_{timestamp}"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # 1. Define Configurations
    # Proposed: All innovations ON
    # Baseline: All innovations OFF (Standard NSGA-II)
    configs = {
        "Baseline": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False"
        ],
        "Proposed (Weak Proto)": [ # 使用 configs/default.yaml 中更新后的 gamma0=0.05
            "--prototype-flags=True", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True"
        ],
        "Proposed (No Proto)": [ # 完全关闭原型交叉，验证仅靠变异和动态精英的效果
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True"
        ],
        "Proposed (Only Coupled)": [ # 仅测试耦合变异，关闭动态精英（使用标准 NSGA-II 选择）
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=False"
        ],
        "Proposed (Only Elite)": [ # 仅测试动态精英，关闭耦合变异（使用标准突变）
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=True"
        ]
    }

    # Run Experiments
    results = []
    
    # We need a common reference point for HV.
    # Let's run first and find bounds, or define fixed ones based on domain knowledge.
    # Yield max ~ 100 => Transformed: 100. Min ~ 0.
    # Cost max ~ 20000 => Transformed: -20000. Min ~ -20000.
    # Loss max ~ 100 => Transformed: -100.
    # Ref point for maximization: should be lower bound of all objectives.
    # Transformed range: Yield [20, 100], Cost [-20000, -2000], Loss [-100, -5]
    # Ref point could be [0, -25000, -150]
    ref_point = np.array([200.0, -1000.0, -1.0]) # A bit loose upper bound for maximization (original min lower bound)
    # Actually for maximization, Ref point is the Upper Bound of the objective space dominated.
    # Wait, usually HV is volume between Pareto Front and Reference Point (Nadir point).
    # For Maximization: Ref Point should be WORSE (Lower) than all points.
    # Val 1 (High is good), Val 2 (High is good). Ref point (Low, Low).
    # Transformed Yield (-(-100))=100. Worst yield = 0.
    # Transformed Cost (-20000). Worst cost = -high.
    ref_point = np.array([0.0, -30000.0, -200.0])

    for name, flags in configs.items():
        for seed in args.seeds:
            out_dir = artifacts_root / f"{name}_seed{seed}"
            if not (out_dir / "final_objectives.csv").exists():
                run_experiment(name, flags, out_dir, seed)
            
            # Collect data for plotting
            objs = np.loadtxt(out_dir / "final_objectives.csv", delimiter=",")
            if objs.ndim == 1: objs = objs.reshape(1, -1)
            
            hv = calculate_metrics(out_dir, ref_point)
            print(f"{name} (Seed {seed}) HV: {hv:.2f}")
            
            results.append({
                "Method": name,
                "Seed": seed,
                "Hypervolume": hv,
                "Objectives": objs,
                "Yield": -objs[:, 0], # Convert back to positive yield
                "Cost": objs[:, 1],
                "N_Loss": objs[:, 2]
            })

    # 2. Analyze & Visualize
    df = pd.DataFrame(results)
    
    # Boxplot of Hypervolume
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Hypervolume", palette="Set2")
    plt.title("Hypervolume Comparison (Higher is Better)")
    plt.savefig(artifacts_root / "hv_comparison.png")
    print("Saved HV comparison to hv_comparison.png")

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
