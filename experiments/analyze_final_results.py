
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, spacing, nondominated_mask

def calculate_metrics_summary(artifacts_dir):
    artifacts_path = Path(artifacts_dir)
    results = []
    
    # Define reference point for HV (same as in run_ablation_study.py)
    ref_point_min = np.array([0.0, 30000.0, 200.0])
    
    # 1. Collect all objectives to find global Pareto front for IGD
    all_objs_list = []
    
    # Helper to parse directory names
    # Format: {Method}_seed{Seed}
    # But some methods have underscores. We know the methods we ran.
    known_methods = [
        "Baseline_NSGA-II",
        "External_NSGA-III",
        "Proposed_Method (Dynamic+PCX)"
    ]
    
    # First pass: collect all objectives
    for method in known_methods:
        for seed in [42, 123, 2024, 789, 999]:
            # The folder name might differ slightly if I used spaces, but check ls output
            # "Proposed_Method (Dynamic+PCX)_seed123"
            folder_name = f"{method}_seed{seed}"
            obj_path = artifacts_path / folder_name / "final_objectives.csv"
            
            if obj_path.exists():
                objs = np.loadtxt(obj_path, delimiter=",")
                if objs.ndim == 1: objs = objs.reshape(1, -1)
                all_objs_list.append(objs)
            else:
                print(f"Warning: Missing {obj_path}")

    if not all_objs_list:
        print("No data found!")
        return

    all_objs = np.vstack(all_objs_list)
    global_mask = nondominated_mask(all_objs)
    global_pf = all_objs[global_mask]
    
    print(f"Global Pareto Front size: {len(global_pf)}")
    
    # 2. Calculate metrics for each run
    for method in known_methods:
        for seed in [42, 123, 2024, 789, 999]:
            folder_name = f"{method}_seed{seed}"
            obj_path = artifacts_path / folder_name / "final_objectives.csv"
            
            if obj_path.exists():
                objs = np.loadtxt(obj_path, delimiter=",")
                if objs.ndim == 1: objs = objs.reshape(1, -1)
                
                hv = hypervolume_monte_carlo(objs, ref_point_min)
                igd_val = igd(objs, global_pf)
                sp = spacing(objs)
                
                results.append({
                    "Method": method,
                    "Seed": seed,
                    "HV": hv,
                    "IGD": igd_val,
                    "Spacing": sp
                })

    df = pd.DataFrame(results)
    
    # 3. Print Summary Statistics
    print("\n" + "="*80)
    print(f"{'Method':<35} | {'HV (Mean ± Std)':<20} | {'IGD (Mean ± Std)':<20} | {'Spacing':<20}")
    print("-" * 100)
    
    summary = df.groupby("Method").agg(["mean", "std"])
    
    for method in known_methods:
        if method not in summary.index: continue
        
        hv_mean = summary.loc[method, ("HV", "mean")]
        hv_std = summary.loc[method, ("HV", "std")]
        
        igd_mean = summary.loc[method, ("IGD", "mean")]
        igd_std = summary.loc[method, ("IGD", "std")]
        
        sp_mean = summary.loc[method, ("Spacing", "mean")]
        sp_std = summary.loc[method, ("Spacing", "std")]
        
        print(f"{method:<35} | {hv_mean:.2e} ± {hv_std:.2e} | {igd_mean:.2f} ± {igd_std:.2f}   | {sp_mean:.2f} ± {sp_std:.2f}")

    print("="*80 + "\n")
    
    # 4. Statistical Tests (Mann-Whitney U)
    # Compare Proposed vs Baseline
    # Compare Proposed vs External
    
    proposed_data = df[df["Method"] == "Proposed_Method (Dynamic+PCX)"]
    baseline_data = df[df["Method"] == "Baseline_NSGA-II"]
    external_data = df[df["Method"] == "External_NSGA-III"]
    
    if not proposed_data.empty:
        print("Statistical Significance (Mann-Whitney U Test, p-value):")
        
        if not baseline_data.empty:
            print("\nvs Baseline (NSGA-II):")
            # HV (Greater is better)
            _, p_hv = mannwhitneyu(proposed_data["HV"], baseline_data["HV"], alternative='greater')
            # IGD (Less is better)
            _, p_igd = mannwhitneyu(proposed_data["IGD"], baseline_data["IGD"], alternative='less')
            
            print(f"  HV > Baseline: p={p_hv:.4f} {'*' if p_hv < 0.05 else ''}")
            print(f"  IGD < Baseline: p={p_igd:.4f} {'*' if p_igd < 0.05 else ''}")

        if not external_data.empty:
            print("\nvs External (NSGA-III):")
            # HV
            _, p_hv = mannwhitneyu(proposed_data["HV"], external_data["HV"], alternative='greater')
            # IGD
            _, p_igd = mannwhitneyu(proposed_data["IGD"], external_data["IGD"], alternative='less')
            
            print(f"  HV > NSGA-III: p={p_hv:.4f} {'*' if p_hv < 0.05 else ''}")
            print(f"  IGD < NSGA-III: p={p_igd:.4f} {'*' if p_igd < 0.05 else ''}")

if __name__ == "__main__":
    # Point to the artifacts directory generated
    # I'll use a relative path assuming I run it from project root
    # artifacts/ablation_study_20260303_203804
    
    # Find the latest ablation directory if possible, or hardcode the one we just made
    target_dir = "degree_scheme_2/scheml_2/artifacts/ablation_study_20260303_203804"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        
    calculate_metrics_summary(target_dir)
