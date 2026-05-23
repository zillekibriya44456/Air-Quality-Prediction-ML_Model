import os
import numpy as np
import pandas as pd

def generate_realistic_data(n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    
    # Hidden variable: pollution level from 0 (very clean) to 1 (extremely polluted)
    pollution_level = rng.uniform(0.0, 1.0, size=n_samples)
    
    # Generate PM10 based on pollution level
    # Clean: ~10 µg/m³, Polluted: ~260 µg/m³
    PM10 = (pollution_level * 240.0 + rng.normal(12.0, 6.0, size=n_samples)).clip(5.0, 320.0)
    
    # PM2.5 is physically a fraction of PM10 (typically 40% to 75%)
    # Let's add some variation and small physical noise
    pm_ratio = rng.uniform(0.42, 0.73, size=n_samples)
    PM2_5 = (PM10 * pm_ratio + rng.normal(0.0, 1.5, size=n_samples)).clip(2.0, PM10 * 0.95)
    
    # NO2 (ppb): Clean: ~5 ppb, Polluted: ~120 ppb
    NO2 = (pollution_level * 110.0 + rng.normal(8.0, 4.0, size=n_samples)).clip(1.0, 150.0)
    
    # SO2 (ppb): Clean: ~2 ppb, Polluted: ~60 ppb
    SO2 = (pollution_level * 55.0 + rng.normal(3.0, 2.0, size=n_samples)).clip(0.5, 90.0)
    
    # CO (ppm): Clean: ~0.1 ppm, Polluted: ~6.0 ppm
    CO = (pollution_level * 5.5 + rng.normal(0.2, 0.1, size=n_samples)).clip(0.02, 10.0)
    
    # O3 (ppb): Ozone is secondary, often higher on sunny days (can be high even in mid-pollution)
    O3 = ((1.0 - pollution_level) * 45.0 + pollution_level * 95.0 + rng.normal(20.0, 8.0, size=n_samples)).clip(1.0, 180.0)
    
    # Temperature (°C): Sunny hot stagnant days promote ozone and particulate trapping
    temperature = (20.0 + pollution_level * 15.0 + rng.normal(0.0, 4.0, size=n_samples)).clip(-5.0, 48.0)
    
    # Humidity (%): Higher humidity can increase particulate aggregation/mass
    humidity = (50.0 - pollution_level * 20.0 + rng.normal(15.0, 8.0, size=n_samples)).clip(10.0, 98.0)
    
    # Wind Speed (m/s): Higher wind speeds disperse pollution, leading to clean air
    wind_speed = (8.0 - pollution_level * 7.0 + rng.normal(1.0, 0.8, size=n_samples)).clip(0.2, 18.0)
    
    df = pd.DataFrame({
        "PM10": np.round(PM10, 2),
        "NO2": np.round(NO2, 2),
        "SO2": np.round(SO2, 2),
        "CO": np.round(CO, 3),
        "O3": np.round(O3, 2),
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "wind_speed": np.round(wind_speed, 2),
        "PM2.5": np.round(PM2_5, 2)
    })
    
    return df

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("Generating physically accurate air quality datasets...")
    
    # Regenerate all 10 datasets with different seeds
    for idx in range(1, 11):
        seed = 1000 + idx * 42
        df = generate_realistic_data(120, seed) # 120 rows per file
        out_path = os.path.join(project_dir, f"air_quality_dataset_{idx}.csv")
        df.to_csv(out_path, index=False)
        print(f"Generated {out_path} with {len(df)} rows.")
        
    print("All 10 datasets successfully regenerated!")

if __name__ == "__main__":
    main()
