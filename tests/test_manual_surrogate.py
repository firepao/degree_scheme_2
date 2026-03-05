import sys
from pathlib import Path
import numpy as np
import joblib
import torch

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.append(str(src_path))

from fertopt.models.surrogate import SurrogateManager, SurrogateParams

ARTIFACTS_DIR = project_root / 'artifacts' / 'surrogate_comparison'
MODEL_PATH = ARTIFACTS_DIR / 'best_dl_model.pth'
SCALER_PATH = ARTIFACTS_DIR / 'scaler.joblib'

def test_pytorch_surrogate():
    print("Testing PyTorch Surrogate Integration...")
    
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("Model or Scaler not found. Skipping test.")
        return

    # Load scaler to get input dimension
    scaler = joblib.load(SCALER_PATH)
    # Scaler was fitted on 9 features, but model expects 13.
    # We must manually specify input dim for test based on model checkpoint (13)
    input_dim = 13 
    print(f"Input dimension for model: {input_dim}")
    
    # Define params
    params = SurrogateParams(
        enabled=True,
        update_interval_g=1,
        query_batch_size=5,
        target_objectives=['Yield'], # Assuming target is Yield
        model_num_estimators=100,
        model_learning_rate=0.1,
        seed=42,
        model_type='pytorch',
        model_path=str(MODEL_PATH),
        scaler_path=str(SCALER_PATH)
    )
    
    manager = SurrogateManager(objective_names=['Yield'], params=params)
    
    # Create dummy data
    # X needs to match input_dim
    x_dummy = np.random.rand(10, input_dim)
    y_dummy = np.random.rand(10, 1) * 100 # Dummy yield
    
    # Initialize (should load model)
    print("Initializing manager...")
    manager.initialize(x_dummy, y_dummy)
    
    # Verify model loaded
    if 0 in manager._models:
        print("Model loaded successfully.")
        model = manager._models[0]
        print(f"Model type: {type(model)}")
        # Check if on GPU if available
        if torch.cuda.is_available():
            param = next(model.parameters())
            print(f"Model device: {param.device}")
            if param.device.type == 'cuda':
                print("Model is on CUDA.")
            else:
                print("Model is on CPU.")
    else:
        print("Model NOT loaded.")
        return

    # Check Prediction
    print("Testing prediction...")
    def true_eval_fn(x):
        return np.ones((len(x), 1)) * -1 # Should not be called for target objective if model works
        
    preds = manager.predict_objectives(x_dummy, true_eval_fn)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions sample: {preds[:3]}")
    
    if np.all(preds == -1):
        print("FAIL: Used fallback function instead of model.")
    else:
        print("SUCCESS: Used model for prediction.")

    # Check Active Update (Fine-tuning)
    print("Testing active update (fine-tuning)...")
    # Provide enough data to trigger update logic if any threshold exists
    x_candidates = np.random.rand(20, input_dim)
    
    # active update needs generation index > 0 and % interval == 0
    updated = manager.active_update(1, x_candidates, lambda x: np.random.rand(len(x), 1))
    
    if updated:
        print("Active update trigger returned True.")
    else:
        print("Active update trigger returned False.")

if __name__ == "__main__":
    test_pytorch_surrogate()
