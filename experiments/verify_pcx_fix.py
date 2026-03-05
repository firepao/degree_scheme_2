
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
    from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, spacing, nondominated_mask
except ImportError:
    # Fallback if imports fail (shouldn't happen with PYTHONPATH set)
    print("Warning: Could not import metrics. Functionality will be limited.")

def main():
    seeds = [42, 123, 2024, 789, 999]
    artifacts_root = project_root / "artifacts/verification_pcx_fix"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    configs = {
        "Baseline_NSGA-II": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False",
            "--crossover-method=sbx" 
        ],
        "Module_Fixed_PCX": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False",
            "--crossover-method=pcx"
        ],
        "Module_Coupled_Mutation": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=False",
            "--crossover-method=sbx"
        ],
        "Module_Dynamic_Elite": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=True",
            "--crossover-method=sbx"
        ],
        "Fixed_Ultimate_Model": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=True",
            "--crossover-method=pcx"
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
            
            # Run if not exists
            if not final_obj_path.exists():
                print(f"Running {name} (Seed {seed})...")
                cmd = [
                    sys.executable,
                    str(project_root / "experiments/run_baseline.py"),
                    "--config", str(project_root / "configs/default.yaml"),
                    "--out", str(out_dir),
                    "--seed", str(seed),
                    "--no-timestamp"
                ] + flags
                
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f"Error running {name} seed {seed}: {ret.stderr}")
                    continue
            else:
                print(f"Skipping {name} (Seed {seed}) - already exists")
            
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
                        "Hypervolume": hv,
                        "Yield": -objs[:, 0],
                        "Cost": objs[:, 1],
                        "Objectives": objs
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
    plt.title("Comparison: Baseline vs Fixed Ultimate Model")
    plt.savefig(artifacts_root / "hv_comparison.png")
    print(f"Saved plot to {artifacts_root / 'hv_comparison.png'}")
    
    # Text Report
    print("\nSummary:")
    summary = df.groupby("Method")["Hypervolume"].agg(["mean", "std"])
    print(summary)

if __name__ == "__main__":
    main()
