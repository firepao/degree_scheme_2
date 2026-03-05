import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib  # Added joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time

# --- 配置 ---
DATA_PATH = Path(r"D:\degree_code_scheml_2\scheml_2\data\raw\strawberry_nutrients.csv")
OUTPUT_DIR = Path(r"D:\degree_code_scheml_2\scheml_2\artifacts\surrogate_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# 设定随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 1. 数据预处理 ---
def load_and_preprocess(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 目标变量
    target_col = 'Yield'
    
    # 特征选择 (排除 ID 和 目标)
    drop_cols = ['Name', 'Category_pH', 'N_Ratio', 'P_Ratio', 'K_Ratio'] # N_Ratio等可能是衍生的，先只用原始N/P/K
    feature_cols = [c for c in df.columns if c not in [target_col] + drop_cols]
    
    print(f"Features: {feature_cols}")
    
    # 分离特征和目标
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 处理类别特征
    cat_cols = ['Fertility', 'Photoperiod', 'Soil_Type', 'Season']
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # Label Encoding for Categorical
    for col in cat_cols:
        le = LabelEncoder()
        # Ensure consistent string type
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Scaling Numerical Features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, cat_cols, num_cols, scaler

# --- 2. LightGBM 模型 ---
def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n--- Training LightGBM ---")
    start_time = time.time()
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': SEED,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    train_time = time.time() - start_time
    
    print(f"LightGBM MSE: {mse:.4f}, R2: {r2:.4f}, Time: {train_time:.2f}s")
    return model, y_pred, mse, r2, train_time

# --- 3. Deep Learning 模型 (ResNet for Tabular) ---
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

def train_dl(X_train, y_train, X_test, y_test, device='cpu'):
    print(f"\n--- Training Deep Learning (Tabular ResNet) on {device} ---")
    start_time = time.time()
    
    # 转换为 Tensor
    X_train_tens = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tens = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tens = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tens = torch.tensor(y_test.values, dtype=torch.float32).to(device) # Keep on GPU for metric calc if needed, or CPU later
    
    dataset = TensorDataset(X_train_tens, y_train_tens)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TabularResNet(num_inputs=X_train.shape[1], hidden_dim=128, num_blocks=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    early_stop_count = 0
    patience = 20
    
    epochs = 200
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_tens)
            # Standardize scaler was used, so target is raw yield. 
            # Loss is MSE on raw target.
            val_loss = criterion(val_preds, y_test_tens.unsqueeze(1))
            
        avg_train_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_dl_model.pth")
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_dl_model.pth"))
    model.eval()
    with torch.no_grad():
        y_pred_tens = model(X_test_tens)
        y_pred = y_pred_tens.cpu().numpy().flatten()
        
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    train_time = time.time() - start_time
    
    print(f"Deep Learning MSE: {mse:.4f}, R2: {r2:.4f}, Time: {train_time:.2f}s")
    
    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('DL Training Curve')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "dl_loss_curve.png")
    
    return model, y_pred, mse, r2, train_time

# --- Main ---
def main():
    X_train, X_test, y_train, y_test, _, _, scaler = load_and_preprocess(DATA_PATH)
    
    # Save scaler
    print(f"Saving scaler to {OUTPUT_DIR / 'scaler.joblib'}")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

    # LightGBM
    lgb_model, lgb_pred, lgb_mse, lgb_r2, lgb_time = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Deep Learning (Check helper for GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dl_model, dl_pred, dl_mse, dl_r2, dl_time = train_dl(X_train, y_train, X_test, y_test, device)
    
    # Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, lgb_pred, alpha=0.5, label=f'LightGBM (R2={lgb_r2:.3f})', color='blue')
    plt.scatter(y_test, dl_pred, alpha=0.5, label=f'ResNet-DL (R2={dl_r2:.3f})', color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Proxies Comparison: LightGBM vs Deep Learning')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "prediction_comparison.png")
    
    # Save Report
    with open(OUTPUT_DIR / "comparison_report.txt", "w") as f:
        f.write(f"LightGBM: MSE={lgb_mse:.4f}, R2={lgb_r2:.4f}, Time={lgb_time:.2f}s\n")
        f.write(f"DeepLearning: MSE={dl_mse:.4f}, R2={dl_r2:.4f}, Time={dl_time:.2f}s\n")

if __name__ == "__main__":
    main()
