import sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.fertopt.evaluation.metrics import hypervolume_monte_carlo

ARTIFACTS_DIR = project_root / r"artifacts\ablation_study_20260302_110322"
OUTPUT_REPORT = project_root / r"汇报材料\2026-03-02\ablation_study_summary.md"

def load_objectives(folder):
    obj_path = folder / "final_objectives.csv"
    if not obj_path.exists():
        return None
    return np.loadtxt(obj_path, delimiter=",")

def main():
    if not ARTIFACTS_DIR.exists():
        print(f"Artifacts not found: {ARTIFACTS_DIR}")
        return

    results = []
    
    # Iterate through run folders
    for folder in ARTIFACTS_DIR.glob("*_*"):
        if not folder.is_dir(): continue
        
        name_parts = folder.name.split("_seed")
        if len(name_parts) != 2: continue
        
        method_name = name_parts[0]
        seed = int(name_parts[1])
        
        objs = load_objectives(folder)
        if objs is None: continue
        if objs.ndim == 1: objs = objs.reshape(1, -1)
        
        # Calculate HV
        # Ref Point: [Yield(min, <=0), Cost(min, <=30000), NLoss(min, <=200)]
        ref_point = np.array([0.0, 30000.0, 200.0])
        hv = hypervolume_monte_carlo(objs, ref_point)
        
        # Calculate Average Metrics
        avg_yield = -np.mean(objs[:, 0]) # Convert back to positive
        avg_cost = np.mean(objs[:, 1])
        avg_nloss = np.mean(objs[:, 2])
        
        results.append({
            "Method": method_name,
            "Seed": seed,
            "Hypervolume": hv,
            "Avg_Yield": avg_yield,
            "Avg_Cost": avg_cost,
            "Avg_NLoss": avg_nloss
        })
        
    df = pd.DataFrame(results)
    
    # Calculate Mean & Std per Method
    summary = df.groupby("Method").agg({
        "Hypervolume": ["mean", "std"],
        "Avg_Yield": "mean",
        "Avg_Cost": "mean",
        "Avg_NLoss": "mean"
    }).sort_values(("Hypervolume", "mean"), ascending=False)
    
    print("\n--- Ablation Study Summary ---")
    print(summary)
    
    # Generate Markdown Table
    md_lines = ["\n### 消融实验详细数据 (Ablation Study Results)\n"]
    md_lines.append("| Method (策略) | HV (Mean ± Std) | Yield (Avg) | Cost (Avg) | N_Loss (Avg) |")
    md_lines.append("|---|---|---|---|---|")
    
    for method, row in summary.iterrows():
        hv_mean = row[("Hypervolume", "mean")]
        hv_std = row[("Hypervolume", "std")]
        y_mean = row[("Avg_Yield", "mean")]
        c_mean = row[("Avg_Cost", "mean")]
        n_mean = row[("Avg_NLoss", "mean")]
        
        # Normalize Method Name for display
        display_name = method.replace("Module_", "").replace("_", " ")
        if "Full" in display_name: display_name = "**Full Proposed**"
        if "Baseline" in display_name: display_name = "Baseline (NSGA-II)"
            
        md_lines.append(f"| {display_name} | {hv_mean:.2e} ± {hv_std:.2e} | {y_mean:.1f} | {c_mean:.0f} | {n_mean:.1f} |")
        
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Summary written to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
