import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

class SurrogatePredictor:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.encoders = {}
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        # Load model
        model_path = self.artifacts_dir / "lgb_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        
        # Load encoders
        encoders_path = self.artifacts_dir / "encoders.pkl"
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
            
        # Load feature names
        feature_names_path = self.artifacts_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
             raise FileNotFoundError(f"Feature names not found at {feature_names_path}")

    def predict(self, decision_values, context):
        """
        Predict yield for a single stage given decision variables and context.
        
        Args:
            decision_values (dict): Dictionary with 'Nitrogen', 'Phosphorus', 'Potassium'.
            context (dict): Dictionary with environmental context (e.g., 'Season', 'Temperature'...).
            
        Returns:
            float: Predicted yield.
        """
        # Merge decision values and context
        input_data = {**context, **decision_values}
        
        # Add derived features if needed (N_Ratio, P_Ratio, K_Ratio)
        # Assuming Ratios are roughly percentage or simple ratio based on some total.
        # But in the dataset, N_Ratio/P_Ratio/K_Ratio are features.
        # Let's check the dataset logic. In many datasets, these are ratios of fertilizer components.
        # If we don't have a formula, we might need to infer or use placeholders.
        # Let's assume for now they are constant or derived from NPK.
        # Based on typical NPK fertilizer logic (e.g. 10-10-10), Ratio might be N / (N+P+K) * 100.
        # Let's try to calculate them if missing.
        
        total_fert = input_data.get('Nitrogen', 0) + input_data.get('Phosphorus', 0) + input_data.get('Potassium', 0)
        if total_fert > 0:
            if 'N_Ratio' not in input_data:
                input_data['N_Ratio'] = (input_data['Nitrogen'] / total_fert) * 100
            if 'P_Ratio' not in input_data:
                input_data['P_Ratio'] = (input_data['Phosphorus'] / total_fert) * 100
            if 'K_Ratio' not in input_data:
                input_data['K_Ratio'] = (input_data['Potassium'] / total_fert) * 100
        else:
             # Default to 0 or equal ratio if no fertilizer
             input_data['N_Ratio'] = input_data.get('N_Ratio', 0)
             input_data['P_Ratio'] = input_data.get('P_Ratio', 0)
             input_data['K_Ratio'] = input_data.get('K_Ratio', 0)

        # Prepare DataFrame to ensure correct order and encoding
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unknown labels gracefully (though context should be consistent with training)
                # Map unknown values to -1 or a known class if possible.
                # Here we assume context values are valid.
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                     # Fallback for unseen labels: use first class or similar hack
                     # Ideally, training data should cover all contexts.
                     print(f"Warning: Unseen label in column {col}: {df[col].iloc[0]}")
                     df[col] = 0 # Dummy value

        # Ensure columns are in the correct order
        # Add missing columns with 0 if any
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        
        # Predict
        prediction = self.model.predict(df)
        return float(prediction[0])

    def predict_batch(self, batch_decisions, batch_contexts):
        """
        Predict for a batch of inputs.
        Args:
            batch_decisions (list of dict): List of NPK dicts.
            batch_contexts (list of dict): List of corresponding contexts.
        """
        # Similar logic but vectorized for efficiency would be better.
        # For simplicity, looping or simple DataFrame construction.
        
        data_list = []
        for dec, ctx in zip(batch_decisions, batch_contexts):
            row = {**ctx, **dec}
            # Add Ratios
            total_fert = row.get('Nitrogen', 0) + row.get('Phosphorus', 0) + row.get('Potassium', 0)
            if total_fert > 0:
                if 'N_Ratio' not in row:
                    row['N_Ratio'] = (row['Nitrogen'] / total_fert) * 100
                if 'P_Ratio' not in row:
                    row['P_Ratio'] = (row['Phosphorus'] / total_fert) * 100
                if 'K_Ratio' not in row:
                    row['K_Ratio'] = (row['Potassium'] / total_fert) * 100
            else:
                 row['N_Ratio'] = row.get('N_Ratio', 0)
                 row['P_Ratio'] = row.get('P_Ratio', 0)
                 row['K_Ratio'] = row.get('K_Ratio', 0)
            data_list.append(row)
            
        df = pd.DataFrame(data_list)
        
        # Encode
        for col, encoder in self.encoders.items():
            if col in df.columns:
                 # Use apply to handle potential errors row-wise or just map
                 # For speed, assume valid
                 df[col] = encoder.transform(df[col].astype(str))

        # Reorder
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        
        return self.model.predict(df)

