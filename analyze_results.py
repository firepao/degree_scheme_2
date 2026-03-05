import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Define path to artifacts
artifacts_dir = Path("artifacts")
# Find the latest ablation study directory
dirs = sorted([d for d in artifacts_dir.glob("ablation_study_*") if d.is_dir()])
if not dirs:
    print("No artifacts found.")
    sys.exit(1)

latest_dir = dirs[-1]
print(f"Analyzing artifacts from: {latest_dir}")

# Load results
results = []
ref_point_min = np.array([0.0, 30000.0, 200.0])

# Metrics functions (simplified implementation for quick check)
def hypervolume(objs, ref):
    # Very basic approximation or placeholder if we can't import the library
    # But wait, we can try to import the project modules if we set path
    return 0.0

def calculate_domination(objs):
    # Count how many points in A dominate B
    return 0

# Let's try to import the actual metrics from src
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from fertopt.evaluation.metrics import hypervolume_monte_carlo, igd, spacing, nondominated_mask
    has_metrics = True
except ImportError:
    print("Could not import metrics from src. Using placeholders.")
    has_metrics = False

# Iterate over subdirectories
for d in latest_dir.iterdir():
    if not d.is_dir(): continue
    
    parts = d.name.split("_seed")
    if len(parts) != 2: continue
    
    method_name = parts[0]
    seed = int(parts[1])
    
    obj_path = d / "final_objectives.csv"
    if not obj_path.exists(): continue
    
    try:
        objs = np.loadtxt(obj_path, delimiter=",")
        if objs.ndim == 1: objs = objs.reshape(1, -1)
        
        results.append({
            "Method": method_name,
            "Seed": seed,
            "Objectives": objs,
            "Yield": -objs[:, 0],
            "Cost": objs[:, 1],
            "N_Loss": objs[:, 2]
        })
    except Exception as e:
        print(f"Error reading {d.name}: {e}")

if not results:
    print("No valid results found.")
    sys.exit(1)

# Compute Metrics
all_objs_list = [r["Objectives"] for r in results]
all_objs = np.vstack(all_objs_list)

if has_metrics:
    # Global PF for IGD
    global_mask = nondominated_mask(all_objs)
    global_pf = all_objs[global_mask]
    
    print("\n--- Results Summary ---")
    print(f"{'Method':<25} | {'Seed':<5} | {'HV (e8)':<10} | {'IGD':<10} | {'Spacing':<10}")
    print("-" * 75)
    
    stats = []
    
    for r in results:
        hv = hypervolume_monte_carlo(r["Objectives"], ref_point_min)
        igd_val = igd(r["Objectives"], global_pf)
        sp = spacing(r["Objectives"])
        
        r["Hypervolume"] = hv
        r["IGD"] = igd_val
        r["Spacing"] = sp
        
        print(f"{r['Method']:<25} | {r['Seed']:<5} | {hv/1e8:<10.4f} | {igd_val:<10.4f} | {sp:<10.4f}")
        
        stats.append({
            "Method": r["Method"],
            "Hypervolume": hv,
            "IGD": igd_val,
            "Spacing": sp
        })

    # DataFrame for aggregation
    df = pd.DataFrame(stats)
    agg = df.groupby("Method").agg(["mean", "std"])
    
    print("\n--- Aggregated Statistics ---")
    print(agg)
    
    # Save comparison plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Method", y="Hypervolume")
    plt.title("Hypervolume Comparison (Higher is Better)")
    plt.savefig(latest_dir / "comparison_hv.png")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Method", y="IGD")
    plt.title("IGD Comparison (Lower is Better)")
    plt.savefig(latest_dir / "comparison_igd.png")
    
    print(f"\nPlots saved to {latest_dir}")

