
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import subprocess

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
import os
os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")

from fertopt.evaluation.metrics import hypervolume_monte_carlo

def run_diagnosis():
    print("Starting Dynamic Elite Diagnosis (5 Seeds)...")
    
    seeds = [42, 123, 2024, 789, 999]
    artifacts_root = project_root / "artifacts/diagnosis_elite_20260303"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    
    configs = {
        # "Baseline_NSGA-II": [
        #     "--prototype-flags=False", 
        #     "--coupled-flags=False", 
        #     "--dynamic-elite-flags=False",
        #     "--crossover-method=sbx"
        # ],
        # "Module_Dynamic_Elite": [
        #     "--prototype-flags=False", 
        #     "--coupled-flags=False", 
        #     "--dynamic-elite-flags=True",
        #     "--crossover-method=sbx"
        # ],
        "Module_Fixed_Dynamic_Elite": [
             "--prototype-flags=False", 
             "--coupled-flags=False", 
             "--dynamic-elite-flags=True",
             "--crossover-method=sbx"
        ]
    }
    
    results = []
    
    # Only run seed 42 first for verification
    verification_seeds = [42]

    for name, flags in configs.items():
        for seed in verification_seeds:
            out_dir = artifacts_root / f"{name}_seed{seed}"
            final_obj_path = out_dir / "final_objectives.csv"
            
            if not final_obj_path.exists():
                if not out_dir.exists():
                    out_dir.mkdir(parents=True)
                    
                cmd = [
                    sys.executable,
                    str(project_root / "experiments/run_baseline.py"),
                    "--config", str(project_root / "configs/default.yaml"),
                    "--out", str(out_dir),
                    "--seed", str(seed),
                    "--no-timestamp"
                ] + flags
                
                print(f"Running {name} (Seed {seed})...")
                try:
                    # Remove capture_output=True to see output and avoid pipe deadlocks
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  -> Error: {e}")
            else:
                print(f"Skipping {name} (Seed {seed}) - already exists")
                
            try:
                # Analyze Result
                if final_obj_path.exists():
                    objs = np.loadtxt(final_obj_path, delimiter=",")
                    if objs.ndim == 1: objs = objs.reshape(1, -1)
                    
                    # Calculate HV (minimization)
                    # Yield (-80), Cost (5000), Loss (20)
                    ref_point = np.array([0.0, 30000.0, 200.0])
                    hv = hypervolume_monte_carlo(objs, ref_point)
                    
                    results.append({
                        "Method": name,
                        "Seed": seed,
                        "Hypervolume": hv
                    })
                    print(f"  -> HV: {hv:.2e}")
                else:
                    print(f"  -> Failed: Output file not found")
            except subprocess.CalledProcessError as e:
                print(f"  -> Error: {e}")

    # Report
    if results:
        df = pd.DataFrame(results)
        print("\n=== Diagnosis Report ===")
        print(df.groupby("Method")["Hypervolume"].describe())
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Method", y="Hypervolume")
        plt.title("Baseline vs Dynamic Elite (5 runs)")
        plt.savefig(artifacts_root / "diagnosis_boxplot.png")
        print(f"Plot saved to {artifacts_root / 'diagnosis_boxplot.png'}")

if __name__ == "__main__":
    run_diagnosis()
