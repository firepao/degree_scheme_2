from __future__ import annotations

from typing import Callable, Dict
import numpy as np
import os
import json
from pathlib import Path

# Try importing SurrogatePredictor, handle if dependencies missing
try:
    from ..evaluation.surrogate_predictor import SurrogatePredictor
    HAS_SURROGATE = True
except ImportError:
    HAS_SURROGATE = False


ObjectiveFn = Callable[[np.ndarray], float]


class ObjectiveRegistry:
    def __init__(self) -> None:
        self._funcs: Dict[str, ObjectiveFn] = {}

    def register(self, name: str, fn: ObjectiveFn) -> None:
        self._funcs[name] = fn

    def resolve(self, names: list[str]) -> list[ObjectiveFn]:
        unresolved = [name for name in names if name not in self._funcs]
        if unresolved:
            raise KeyError(f"未注册目标函数: {unresolved}")
        return [self._funcs[name] for name in names]


def build_default_registry(project_root: str = None) -> ObjectiveRegistry:
    registry = ObjectiveRegistry()

    # Load Surrogate Model if available
    predictor = None
    seasonal_contexts = {}
    
    if HAS_SURROGATE and project_root:
        artifacts_dir = Path(project_root) / "artifacts/surrogate_model"
        context_path = artifacts_dir / "seasonal_contexts.json"
        
        if (artifacts_dir / "lgb_model.pkl").exists() and context_path.exists():
            try:
                predictor = SurrogatePredictor(artifacts_dir)
                with open(context_path, 'r') as f:
                    seasonal_contexts = json.load(f)
                print("已加载代理模型和季节性上下文。")
            except Exception as e:
                print(f"加载代理模型失败: {e}")
                predictor = None
        else:
            print("未找到代理模型文件，使用模拟目标函数。")

    # Define objectives using surrogate or fallback
    
    def yield_obj(x: np.ndarray) -> float:
        # x is [N1, P1, K1, N2, P2, K2, ..., Nn, Pn, Kn]
        # Calculate number of stages dynamically
        n_stages = x.size // 3
        
        if predictor and seasonal_contexts:
            total_yield = 0.0
            # Define available seasons for cycling
            season_cycle = ["Spring", "Summer", "Fall", "Winter"]
            
            try:
                # Reshape x into (n_stages, 3)
                stages = x.reshape(n_stages, 3)
                
                for i in range(n_stages):
                    # Cycle through available seasons
                    season_key = season_cycle[i % len(season_cycle)]
                    
                    if season_key in seasonal_contexts:
                        context = seasonal_contexts[season_key]
                        # Decision vars
                        decision = {
                            'Nitrogen': stages[i, 0],
                            'Phosphorus': stages[i, 1],
                            'Potassium': stages[i, 2]
                        }
                        # Predict
                        y = predictor.predict(decision, context)
                        total_yield += y
                    else:
                        print(f"Warning: Season {season_key} context missing.")
                        
                return -float(total_yield) # Minimize negative yield = Maximize Yield
            except Exception as e:
                print(f"Objectives Error: {e}")
                return 0.0
        
        else:
            # Fallback Simulation
            try:
                stage_values = x.reshape(-1, 3)
                npk_balance_penalty = np.mean((stage_values[:, 0] - stage_values[:, 1]) ** 2) * 0.001
                productivity = np.sum(np.sqrt(np.clip(stage_values, 0.0, None))) - npk_balance_penalty
                return -float(productivity)
            except Exception:
                return 0.0

    def cost_obj(x: np.ndarray) -> float:
        price = np.array([5.2, 6.8, 4.5], dtype=float) # Price for N, P, K
        try:
             stage_values = x.reshape(-1, 3)
             return float(np.sum(stage_values @ price))
        except ValueError:
             return 0.0

    def nitrogen_loss_obj(x: np.ndarray) -> float:
        try:
            stage_values = x.reshape(-1, 3)
            n = stage_values[:, 0]
            p = stage_values[:, 1]
            k = stage_values[:, 2]
            loss = 0.06 * np.sum(n) + 0.01 * np.sum(np.maximum(n - (p + k) * 0.5, 0.0))
            return float(loss)
        except ValueError:
             return 0.0

    registry.register("yield", yield_obj)
    registry.register("cost", cost_obj)
    registry.register("nitrogen_loss", nitrogen_loss_obj)
    return registry
