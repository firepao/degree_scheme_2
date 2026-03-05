from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
from pathlib import Path

import numpy as np
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Mock for definition time if torch missing
    class nn:
        Module = object



@dataclass(slots=True)
class SurrogateParams:
    enabled: bool
    update_interval_g: int
    query_batch_size: int
    target_objectives: list[str]
    model_num_estimators: int
    model_learning_rate: float
    seed: int
    # New params for PyTorch
    model_type: str = 'lightgbm'  # 'lightgbm' or 'pytorch'
    model_path: str = ''
    scaler_path: str = ''


# --- PyTorch Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class TabularResNet(nn.Module):
    def __init__(self, num_inputs, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class SurrogateManager:
    def __init__(self, objective_names: list[str], params: SurrogateParams) -> None:
        self.objective_names = objective_names
        self.params = params
        self._target_indices = [
            idx for idx, name in enumerate(objective_names) if name in set(params.target_objectives)
        ]

        self._models: dict[int, Any] = {}
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        
        # PyTorch specific
        self._scaler = None
        self._device = None
        
        if self.params.model_type == 'pytorch' and not HAS_TORCH:
             raise ImportError("Surrogate model_type='pytorch' but torch is not installed.")

    @property
    def is_enabled(self) -> bool:
        return self.params.enabled

    def initialize(self, x: np.ndarray, y_true: np.ndarray) -> None:
        if not self.is_enabled:
            return
        
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_y = np.asarray(y_true, dtype=float).copy()
        
        if self.params.model_type == 'pytorch':
            # Initialize/Load resources
            self._load_pytorch_resources(x.shape[1])
            # Only fine-tune if we have enough initial data, otherwise just load pretrained
            if self._train_x.shape[0] > 10: 
                self._fit_models()
        else:
            self._fit_models()

    def _load_pytorch_resources(self, input_dim: int):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler
        if self.params.scaler_path and Path(self.params.scaler_path).exists():
            print(f"Loading scaler from {self.params.scaler_path}")
            self._scaler = joblib.load(self.params.scaler_path)
        else:
            print(f"Warning: Scaler not found at {self.params.scaler_path}, creating new scaler.")
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            self._scaler.fit(self._train_x) # Fit on initial data if no pretrained scaler

        # Load Models
        for idx in self._target_indices:
            # Note: num_blocks etc should match the saved model. Assuming default structure here.
            model = TabularResNet(num_inputs=input_dim, hidden_dim=128, num_blocks=3).to(self._device)
            if self.params.model_path and Path(self.params.model_path).exists():
                try:
                    state_dict = torch.load(self.params.model_path, map_location=self._device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    print(f"Loaded PyTorch model from {self.params.model_path}")
                except Exception as e:
                    print(f"Failed to load PyTorch model from {self.params.model_path}: {e}")
            else:
                print(f"Warning: PyTorch model not found at {self.params.model_path}, initializing randomly.")
            
            self._models[idx] = model

    def predict_objectives(self, x: np.ndarray, true_eval_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.is_enabled or self._train_x is None or self._train_y is None or not self._models:
            return true_eval_fn(x)

        if self.params.model_type == 'pytorch':
             return self._predict_pytorch(x, true_eval_fn)

        # LightGBM / Default Path
        y_pred = np.zeros((x.shape[0], len(self.objective_names)), dtype=float)

        fallback_indices = [idx for idx in range(len(self.objective_names)) if idx not in self._target_indices]
        if fallback_indices:
            y_true_part = true_eval_fn(x)
            y_pred[:, fallback_indices] = y_true_part[:, fallback_indices]

        for idx in self._target_indices:
            model = self._models.get(idx)
            if model is None:
                y_true_part = true_eval_fn(x)
                y_pred[:, idx] = y_true_part[:, idx]
            else:
                y_pred[:, idx] = model.predict(x)

        return y_pred

    def _predict_pytorch(self, x: np.ndarray, true_eval_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        y_pred = np.zeros((x.shape[0], len(self.objective_names)), dtype=float)
        
        # Fill non-target objectives with true values
        fallback_indices = [idx for idx in range(len(self.objective_names)) if idx not in self._target_indices]
        if fallback_indices:
            y_true_part = true_eval_fn(x)
            y_pred[:, fallback_indices] = y_true_part[:, fallback_indices]

        # Prepare input
        x_scaled = x.copy() # Avoid modifying original
        if self._scaler:
            try:
                if hasattr(self._scaler, 'n_features_in_') and self._scaler.n_features_in_ < x.shape[1]:
                    # Assume scaler applies to the first N features (numerical)
                    # This is a heuristic based on common pattern: num cols first, then cats.
                    # Ideally, we should pass column indices.
                    n_s = self._scaler.n_features_in_
                    x_scaled[:, :n_s] = self._scaler.transform(x[:, :n_s])
                else:
                    x_scaled = self._scaler.transform(x)
            except Exception as e:
                # Fallback
                print(f"Scaling failed: {e}")
        
        x_tens = torch.tensor(x_scaled, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            for idx in self._target_indices:
                model = self._models.get(idx)
                if model:
                    model.eval()
                    pred_tens = model(x_tens)
                    y_pred[:, idx] = pred_tens.cpu().numpy().flatten()
                else:
                    y_true_part = true_eval_fn(x)
                    y_pred[:, idx] = y_true_part[:, idx]
        
        return y_pred

    def active_update(
        self,
        generation_index: int,
        x_candidates: np.ndarray,
        true_eval_fn: Callable[[np.ndarray], np.ndarray],
    ) -> bool:
        if not self.is_enabled:
            return False
        if generation_index <= 0:
            return False
        if generation_index % max(1, self.params.update_interval_g) != 0:
            return False
        if self._train_x is None or self._train_y is None:
            return False

        x_candidates = np.asarray(x_candidates, dtype=float)
        if x_candidates.shape[0] == 0:
            return False

        pick = min(self.params.query_batch_size, x_candidates.shape[0])
        selected_idx = self._select_query_indices(x_candidates, pick)
        x_query = x_candidates[selected_idx]
        y_query = true_eval_fn(x_query)

        self._train_x = np.vstack([self._train_x, x_query])
        self._train_y = np.vstack([self._train_y, y_query])
        
        self._fit_models() # Retrain/Finetune
        return True

    def training_size(self) -> int:
        return 0 if self._train_x is None else int(self._train_x.shape[0])

    def _fit_models(self) -> None:
        if self._train_x is None or self._train_y is None:
            return
        if not self._target_indices:
            return

        if self.params.model_type == 'pytorch':
            self._fit_pytorch()
        else:
            self._fit_lightgbm()

    def _fit_lightgbm(self) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "surrogate.enabled=True 但未安装 lightgbm，请先安装 lightgbm。"
            ) from exc

        self._models = {}
        for idx in self._target_indices:
            model = LGBMRegressor(
                n_estimators=self.params.model_num_estimators,
                learning_rate=self.params.model_learning_rate,
                random_state=self.params.seed + idx,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(self._train_x, self._train_y[:, idx])
            self._models[idx] = model

    def _fit_pytorch(self) -> None:
        # Fine-tune the existing models on _train_x, _train_y
        X = self._train_x
        if self._scaler:
            try:
                X = self._scaler.transform(X) # Use transform, not fit_transform to keep consistent with pretrained
            except:
                pass 
        
        X_tens = torch.tensor(X, dtype=torch.float32).to(self._device)
        
        epochs = 10 # Simple fine-tuning
        
        for idx in self._target_indices:
            y = self._train_y[:, idx]
            y_tens = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self._device)
            
            model = self._models.get(idx)
            if model is None:
                continue 
                
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Lower LR for fine-tuning
            criterion = nn.MSELoss()
            
            dataset = TensorDataset(X_tens, y_tens)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for _ in range(epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()

    def _select_query_indices(self, x_candidates: np.ndarray, pick: int) -> np.ndarray:
        if self._train_x is None or self._train_x.shape[0] == 0:
            return np.arange(pick)

        dist = np.linalg.norm(
            x_candidates[:, None, :] - self._train_x[None, :, :],
            axis=2,
        )
        nearest = np.min(dist, axis=1)
        return np.argsort(-nearest)[:pick]
