import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SurrogateModelTrainer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.encoders = {}
        self.feature_names = []
        self.categorical_features = ['Season', 'Soil_Type', 'Photoperiod', 'Category_pH', 'Fertility']
        self.target_col = 'Yield'
        # Drop columns that are not useful or identifiers
        self.drop_cols = ['Name', 'Random', 'Class'] # Check if 'Random' or 'Class' exist based on previous exploration, 'Name' was in sample.

    def load_data(self):
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Clean data: drop ID columns if they exist
        for col in self.drop_cols:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
        
        print(f"Data shape: {self.df.shape}")
        
    def preprocess(self):
        print("Preprocessing data...")
        # Handle categorical variables
        for col in self.categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                print(f"Encoded {col} with {len(le.classes_)} classes")
            else:
                print(f"Warning: Categorical column {col} not found in data")

        # Define features X and target y
        self.y = self.df[self.target_col]
        self.X = self.df.drop(columns=[self.target_col])
        self.feature_names = self.X.columns.tolist()
        
        # Save feature names for inference alignment
        with open(self.output_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
            
        print(f"Features: {self.feature_names}")

    def train(self):
        print("Training LightGBM model...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': ['rmse', 'l2'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "="*30)
        print("Model Evaluation")
        print("="*30)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump({"mse": mse, "rmse": np.sqrt(mse), "r2": r2}, f)

    def save_artifacts(self):
        print(f"Saving artifacts to {self.output_dir}")
        # Save model
        joblib.dump(self.model, self.output_dir / "lgb_model.pkl")
        # Save model as text for portability (optional)
        self.model.save_model(str(self.output_dir / "lgb_model.txt"))
        
        # Save encoders
        joblib.dump(self.encoders, self.output_dir / "encoders.pkl")
        
        print("Artifacts saved successfully.")

    def plot_importance(self):
        print("Plotting feature importance...")
        ax = lgb.plot_importance(self.model, max_num_features=15, figsize=(10, 8), title="Surrogate Model Feature Importance")
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png")
        print(f"Feature importance plot saved to {self.output_dir / 'feature_importance.png'}")

if __name__ == "__main__":
    # Determine paths
    current_dir = Path(os.getcwd())
    data_path = current_dir / "data/raw/strawberry_nutrients.csv"
    output_dir = current_dir / "artifacts/surrogate_model"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        exit(1)
        
    trainer = SurrogateModelTrainer(data_path, output_dir)
    trainer.load_data()
    trainer.preprocess()
    trainer.train()
    trainer.save_artifacts()
    trainer.plot_importance()
