import os
import glob
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    # Find all air quality dataset files
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_files = glob.glob(os.path.join(project_dir, "air_quality_dataset_*.csv"))
    
    print(f"Found {len(dataset_files)} dataset files to combine.")
    
    all_dfs = []
    required_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'temperature', 'humidity', 'wind_speed', 'PM2.5']
    
    for f in dataset_files:
        try:
            df = pd.read_csv(f)
            # Standardize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            
            # Check if all required columns are present
            if all(col in df.columns for col in required_cols):
                all_dfs.append(df[required_cols])
                print(f"Loaded {os.path.basename(f)} with {len(df)} rows.")
            else:
                missing = [col for col in required_cols if col not in df.columns]
                print(f"Skipping {os.path.basename(f)} due to missing columns: {missing}")
        except Exception as e:
            print(f"Error loading {os.path.basename(f)}: {e}")
            
    if not all_dfs:
        print("No valid datasets found! Checking for any individual air quality CSV files...")
        # Fallback to check other csv files like industrial/urban if no dataset_* files match
        for f in glob.glob(os.path.join(project_dir, "*.csv")):
            try:
                df = pd.read_csv(f)
                df.columns = [c.strip() for c in df.columns]
                if all(col in df.columns for col in required_cols):
                    all_dfs.append(df[required_cols])
                    print(f"Loaded fallback file {os.path.basename(f)} with {len(df)} rows.")
            except Exception as e:
                pass

    if not all_dfs:
        raise ValueError("Could not find any valid CSV files with the required columns.")

    # Combine all dataframes
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.dropna()
    print(f"Total combined dataset rows: {len(full_df)}")
    
    # Features and target split
    X = full_df[['PM10', 'NO2', 'SO2', 'CO', 'O3', 'temperature', 'humidity', 'wind_speed']]
    y = full_df['PM2.5']
    
    # Save statistics of the dataset
    stats = {
        "count": len(full_df),
        "PM10": {"mean": float(X['PM10'].mean()), "min": float(X['PM10'].min()), "max": float(X['PM10'].max())},
        "NO2": {"mean": float(X['NO2'].mean()), "min": float(X['NO2'].min()), "max": float(X['NO2'].max())},
        "SO2": {"mean": float(X['SO2'].mean()), "min": float(X['SO2'].min()), "max": float(X['SO2'].max())},
        "CO": {"mean": float(X['CO'].mean()), "min": float(X['CO'].min()), "max": float(X['CO'].max())},
        "O3": {"mean": float(X['O3'].mean()), "min": float(X['O3'].min()), "max": float(X['O3'].max())},
        "temperature": {"mean": float(X['temperature'].mean()), "min": float(X['temperature'].min()), "max": float(X['temperature'].max())},
        "humidity": {"mean": float(X['humidity'].mean()), "min": float(X['humidity'].min()), "max": float(X['humidity'].max())},
        "wind_speed": {"mean": float(X['wind_speed'].mean()), "min": float(X['wind_speed'].min()), "max": float(X['wind_speed'].max())},
        "PM2_5": {"mean": float(y.mean()), "min": float(y.min()), "max": float(y.max())}
    }
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
        "linear_regression": LinearRegression()
    }
    
    metrics = {}
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict & Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Save model metrics
        metrics[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }
        
        # Save model to disk
        model_path = os.path.join(models_dir, f"{name}_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")
        
    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save stats and metrics metadata
    metadata = {
        "metrics": metrics,
        "stats": stats
    }
    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved training metadata to {metadata_path}")
    print("Training complete successfully!")

if __name__ == "__main__":
    train()
