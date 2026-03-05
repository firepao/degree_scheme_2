
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Import metrics
try:
    from fertopt.evaluation.metrics import hypervolume_monte_carlo
except ImportError:
    print("Warning: Could not import metrics.")

def main():
    seeds = [42, 123, 2024, 789, 999]
    artifacts_root = project_root / "artifacts/final_ultimate_benchmark_v2"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    configs = {
        "Baseline_NSGA-II": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False",
            "--crossover-method=sbx" 
        ],
        "Ultimate_Full": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True",
            "--crossover-method=pcx"
        ],
        "Ultimate_PCX_Only": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=True",
            "--crossover-method=pcx"
        ],
        "Ultimate_Coupled_Only": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True",
            "--crossover-method=sbx"
        ]
    }

    results = []
    
    # Reference Point for Hypervolume (Minimization)
    # Yield [-100, -20], Cost [2000, 30000], Loss [5, 200]
    ref_point_min = np.array([0.0, 30000.0, 200.0])

    for name, flags in configs.items():
        for seed in seeds:
            out_dir = artifacts_root / f"{name}_seed{seed}"
            final_obj_path = out_dir / "final_objectives.csv"
            
            # Always run fresh for final benchmark
            if out_dir.exists():
                import shutil
                shutil.rmtree(out_dir)
            
            print(f"Running {name} (Seed {seed})...")
            cmd = [
                sys.executable,
                str(project_root / "experiments/run_baseline.py"),
                "--config", str(project_root / "configs/default.yaml"),
                "--out", str(out_dir),
                "--seed", str(seed),
                "--no-timestamp"
            ] + flags
            
            # Use alpha0=0.5 for Ultimate Model via config override if needed
            # But we set it in default.yaml already.
            
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"Error running {name} seed {seed}: {ret.stderr}")
                continue
            
            # Collect Results
            if final_obj_path.exists():
                try:
                    objs = np.loadtxt(final_obj_path, delimiter=",")
                    if objs.ndim == 1: objs = objs.reshape(1, -1)
                    
                    # Calculate HV
                    hv = hypervolume_monte_carlo(objs, ref_point_min)
                    
                    results.append({
                        "Method": name,
                        "Seed": seed,
                        "Hypervolume": hv
                    })
                    print(f"  -> HV: {hv:.2e}")
                except Exception as e:
                    print(f"Error analyzing {out_dir}: {e}")

    if not results:
        print("No results.")
        return

    # Visualization
    df = pd.DataFrame(results)
    
    # HV Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Method", y="Hypervolume")
    plt.title("Final Comparison: Baseline vs Ultimate Model")
    plt.savefig(artifacts_root / "final_hv_comparison.png")
    print(f"Saved plot to {artifacts_root / 'final_hv_comparison.png'}")
    
    # Text Report
    print("\nFinal Summary:")
    summary = df.groupby("Method")["Hypervolume"].agg(["mean", "std", "min", "max"])
    print(summary)
    
    # Check if Ultimate beats Baseline
    means = df.groupby("Method")["Hypervolume"].mean()
    if means["Ultimate_Model"] > means["Baseline_NSGA-II"]:
        print("\nSUCCESS: Ultimate Model outperforms Baseline on average!")
    else:
        print("\nFAILURE: Ultimate Model does not outperform Baseline.")

if __name__ == "__main__":
    main()
