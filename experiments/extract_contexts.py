import pandas as pd
import json
from pathlib import Path

def extract_seasonal_contexts(data_path, output_path):
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Define seasons of interest
    target_seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    
    # Features that are context (not decision variables and not yield)
    context_features = [
        'Fertility', 'Photoperiod', 'Temperature', 'Rainfall', 'pH', 
        'Light_Hours', 'Light_Intensity', 'Rh', 'Category_pH', 'Soil_Type'
    ]
    # Decision variables (we will replace these during optimization, but need placeholders)
    decision_vars = ['Nitrogen', 'Phosphorus', 'Potassium', 'N_Ratio', 'P_Ratio', 'K_Ratio']
    
    contexts = {}
    
    for season in target_seasons:
        season_df = df[df['Season'] == season]
        if season_df.empty:
            print(f"Warning: No data for season {season}")
            continue
            
        # For numerical context, take mean
        numerical_cols = season_df[context_features].select_dtypes(include=['number']).columns
        # For categorical context, take mode (most frequent)
        categorical_cols = season_df[context_features].select_dtypes(exclude=['number']).columns
        
        ctx = {}
        for col in numerical_cols:
            ctx[col] = float(season_df[col].mean())
        for col in categorical_cols:
            ctx[col] = season_df[col].mode()[0]
            
        # Add Season explicitly
        ctx['Season'] = season
        
        contexts[season] = ctx
        print(f"Computed context for {season}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(contexts, f, indent=2)
    print(f"Saved seasonal contexts to {output_path}")

if __name__ == "__main__":
    extract_seasonal_contexts(
        "data/raw/strawberry_nutrients.csv",
        "artifacts/surrogate_model/seasonal_contexts.json"
    )
