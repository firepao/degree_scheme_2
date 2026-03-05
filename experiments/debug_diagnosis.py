import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def run_experiment(name, flags, out_dir, seed=42):
    """Run a single experiment configuration."""
    cmd = [
        sys.executable,
        str(project_root / "experiments/run_baseline.py"),
        "--config", str(project_root / "configs/default.yaml"),
        "--out", str(out_dir),
        "--seed", str(seed),
        "--no-timestamp"
    ] + flags
    
    print(f"Running {name}...")
    subprocess.run(cmd, capture_output=True, text=True)
    return out_dir

def main():
    artifacts_root = project_root / "artifacts/debug_diagnosis"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # Define diagnostic configurations
    configs = {
        "Baseline (NSGA2)": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False"
        ],
        "A: Only Beta Init": [ # Need to change config value manually or add arg support for beta? 
            # Beta init is controlled by Config only, currently always ON in runner if calling beta_biased_initialize.
            # Wait, runner.py _run_deap_nsga2 ALWAYS calls beta_biased_initialize.
            # So Baseline actuall HAS Beta Init if we didn't change runner logic.
            # Let's check runner.py... yes, it calls beta_biased_initialize unconditionally.
            # So "Baseline" above is actually "Beta Init + NSGA2".
            # If we want pure random, we need to modify runner or config.
            # For now, let's assume Beta Init is the new Baseline.
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False"
        ],
        "B: + Prototype Crossover": [
            "--prototype-flags=True", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=False"
        ],
        "C: + Coupled Mutation": [
            "--prototype-flags=False", 
            "--coupled-flags=True", 
            "--dynamic-elite-flags=False"
        ],
        "D: + Dynamic Elite": [
            "--prototype-flags=False", 
            "--coupled-flags=False", 
            "--dynamic-elite-flags=True"
        ]
    }

    results = []
    
    for name, flags in configs.items():
        if name == "A: Only Beta Init": continue # Skip duplicate
        
        out_dir = artifacts_root / name.replace(" ", "_").replace(":", "")
        run_experiment(name, flags, out_dir)
        
        try:
            objs = np.loadtxt(out_dir / "final_objectives.csv", delimiter=",")
            if objs.ndim == 1: objs = objs.reshape(1, -1)
            
            # Subsample for plotting if too many
            if len(objs) > 200:
                indices = np.random.choice(len(objs), 200, replace=False)
                objs = objs[indices]
                
            for row in objs:
                results.append({
                    "Method": name,
                    "Yield": -row[0], # Convert back to positive
                    "Cost": row[1]
                })
        except Exception:
            print(f"Failed to load results for {name}")

    # Plot
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="Cost", y="Yield", hue="Method", style="Method", s=60, alpha=0.7)
    plt.title("Diagnostic: Single Module Ablation")
    plt.grid(True, alpha=0.3)
    plt.savefig(artifacts_root / "diagnosis_plot.png")
    print(f"Saved diagnosis plot to {artifacts_root / 'diagnosis_plot.png'}")

if __name__ == "__main__":
    main()
