# Final Optimization Report

## Summary
We successfully diagnosed and fixed the major performance bottlenecks in the "Ultimate Model". The initial implementation suffered from critical bugs in the PCX Crossover and Dynamic Elite Selection operators, causing it to significantly underperform the Baseline NSGA-II (~8.17e8 vs 8.67e8 HV).

After applying fixes, the Ultimate Model now achieves performance comparable to the Baseline (~8.66e8 HV), effectively resolving the regression.

## Key Fixes

### 1. PCX Crossover Fix
- **Issue**: The noise scaling was calculated based on the global domain bounds (`upper - lower`), which prevented fine-grained convergence in later generations.
- **Fix**: Updated `pcx_crossover` in `src/fertopt/operators/crossover.py` to scale noise based on the **distance between parents** (`dist_ab`). This allows the crossover to be explorative early on and exploitative later.
- **Verification**: Verified in `experiments/verify_pcx_fix.py` where PCX module achieved parity with Baseline.

### 2. Dynamic Elite Selection Fix
- **Issue**: 
    1. Lack of normalization allowed large-scale objectives (Yield) to dominate diversity calculations.
    2. The initial KNN-based sparsity metric lacked "Boundary Preservation", causing the Pareto front to shrink compared to NSGA-II's Crowding Distance.
- **Fix**: 
    1. Implemented **Local Normalization** (per front).
    2. Rewrote the selection logic in `src/fertopt/operators/selection.py` to use **Standard Crowding Distance** for the objective space component (ensuring boundary preservation).
    3. Added a **Decision Space Sparsity** component that can be mixed in via `alpha_t`.
- **Verification**: Verified in `experiments/verify_selection_fix.py`. Performance improved from ~8.17e8 to ~8.66e8 (Baseline level).

## Benchmark Results (5 Seeds)

| Method | Mean HV | Std Dev | Status |
| :--- | :--- | :--- | :--- |
| **Baseline NSGA-II** | 8.68e8 | 3.05e6 | Reference |
| **Ultimate (Full)** | 8.66e8 | 4.47e6 | **Fixed** (Matched) |
| **Ultimate (Coupled Only)** | 8.67e8 | 3.57e6 | **Fixed** (Matched) |

## Conclusion
The "Ultimate Model" is no longer broken. The performance regression has been eliminated. While it does not strictly *outperform* the Baseline on this specific benchmark configuration, it is now a robust alternative that includes decision space diversity mechanisms (Coupled Mutation, Dynamic Elite) which may provide benefits in other problem instances or larger runs.
