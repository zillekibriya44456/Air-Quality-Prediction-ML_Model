from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
import urllib.request

app = Flask(__name__)

# Base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Load models, scaler, and metadata
models = {}
scaler = None
metadata = {}

try:
    # Load scaling object
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Load ML models
    for name in ["random_forest", "gradient_boosting", "linear_regression"]:
        path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    # Load metadata (metrics and stats)
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
except Exception as e:
    print(f"Error loading models or metadata: {e}")


# US EPA Breakpoint tables
# Format: (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ],
    "PM10": [
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500)
    ],
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ],
    "NO2": [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 1649.0, 301, 400),
        (1650.0, 2049.0, 401, 500)
    ],
    "SO2": [
        (0.0, 35.0, 0, 50),
        (36.0, 75.0, 51, 100),
        (76.0, 185.0, 101, 150),
        (186.0, 304.0, 151, 200),
        (305.0, 604.0, 201, 300),
        (605.0, 804.0, 301, 400),
        (805.0, 1004.0, 401, 500)
    ],
    "O3": [
        (0.0, 54.0, 0, 50),
        (55.0, 70.0, 51, 100),
        (71.0, 85.0, 101, 150),
        (86.0, 105.0, 151, 200),
        (106.0, 200.0, 201, 300),
        (201.0, 400.0, 301, 500)
    ]
}

def calculate_aqi_subindex(value, pollutant):
    if pollutant not in BREAKPOINTS:
        return 0
    
    # Cap value to max breakpoint if it exceeds
    bp = BREAKPOINTS[pollutant]
    max_val = bp[-1][1]
    if value >= max_val:
        return bp[-1][3]
        
    for c_low, c_high, i_low, i_high in bp:
        if c_low <= value <= c_high:
            # Linear interpolation
            aqi = ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
            return round(aqi)
    return 0

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400", "Air quality is satisfactory, and air pollution poses little or no risk.", "Enjoy outdoor activities. No special precautions needed."
    elif aqi <= 100:
        return "Moderate", "#ffff00", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.", "Sensitive individuals should consider reducing prolonged or heavy outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects. The general public is less likely to be affected.", "People with lung/heart disease, older adults, and children should reduce outdoor activities."
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.", "Everyone should wear masks outside, avoid high intensity workouts outdoors, and close windows."
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "Health alert: The risk of health effects is increased for everyone.", "Avoid all physical activity outdoors. Keep air purifiers running inside."
    else:
        return "Hazardous", "#7e0023", "Health warning of emergency conditions: everyone is more likely to be affected.", "Remain indoors. Keep windows closed and run air filters on high. Wear N95 masks if forced to go out."

def perform_prediction(features_dict, model_name="random_forest"):
    # Check if we have models loaded, otherwise fallback
    pm10 = float(features_dict.get('PM10', 0))
    no2 = float(features_dict.get('NO2', 0))
    so2 = float(features_dict.get('SO2', 0))
    co = float(features_dict.get('CO', 0))
    o3 = float(features_dict.get('O3', 0))
    temp = float(features_dict.get('temperature', 0))
    humidity = float(features_dict.get('humidity', 0))
    wind_speed = float(features_dict.get('wind_speed', 0))
    
    # Check if user passed manual PM2.5 (from PM25 or PM2.5 keys in features_dict)
    user_pm25 = features_dict.get('PM25') or features_dict.get('PM2.5')
    is_manual = False
    predicted_pm25 = None
    if user_pm25 is not None and str(user_pm25).strip() != "":
        try:
            predicted_pm25 = float(user_pm25)
            is_manual = True
        except ValueError:
            pass
            
    # 1. Predict PM2.5 using ML model if available
    if predicted_pm25 is None:
        if model_name in models and scaler is not None:
            try:
                # Inputs: PM10, NO2, SO2, CO, O3, temperature, humidity, wind_speed
                input_df = pd.DataFrame([[pm10, no2, so2, co, o3, temp, humidity, wind_speed]], 
                                        columns=['PM10', 'NO2', 'SO2', 'CO', 'O3', 'temperature', 'humidity', 'wind_speed'])
                input_scaled = scaler.transform(input_df)
                predicted_pm25 = float(models[model_name].predict(input_scaled)[0])
            except Exception as e:
                print(f"Error predicting with ML model: {e}")
                
        # Fallback PM2.5 calculation if no model exists or failed
        if predicted_pm25 is None:
            # Physically realistic fallback calculation based on the new physical relationship:
            # PM2.5 is typically around 55% of PM10 plus some combustion effects (CO, NO2) and wind dispersion
            predicted_pm25 = (pm10 * 0.55) + (no2 * 0.15) + (co * 2.5) - (wind_speed * 0.3)
        
    predicted_pm25 = max(0.1, round(predicted_pm25, 2))
    
    # 2. Compute individual pollutant sub-indices
    sub_indices = {
        "PM2.5": calculate_aqi_subindex(predicted_pm25, "PM2.5"),
        "PM10": calculate_aqi_subindex(pm10, "PM10"),
        "NO2": calculate_aqi_subindex(no2, "NO2"),
        "SO2": calculate_aqi_subindex(so2, "SO2"),
        "CO": calculate_aqi_subindex(co, "CO"),
        "O3": calculate_aqi_subindex(o3, "O3")
    }
    
    # 3. Overall AQI is the maximum of individual sub-indices
    overall_aqi = max(sub_indices.values())
    
    # 4. Determine dominant pollutant
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    if overall_aqi == 0:
        dominant_pollutant = "None"
        
    # 5. Get health advice and styling parameters
    category, color, description, advice = get_aqi_category(overall_aqi)
    
    return {
        "predicted_pm25": predicted_pm25,
        "is_manual_pm25": is_manual,
        "overall_aqi": overall_aqi,
        "sub_indices": sub_indices,
        "dominant_pollutant": dominant_pollutant,
        "category": category,
        "color": color,
        "description": description,
        "advice": advice,
        "inputs": {
            "PM25": user_pm25 if is_manual else "",
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed
        },
        "model_used": model_name
    }

@app.route('/')
def home():
    # Pass metadata (metrics and training stats) to render in dashboard
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model_name', 'random_forest')
        result = perform_prediction(request.form, model_name)
        
        # Prepare prediction text for basic display (backward compatibility)
        prediction_text = f"Predicted AQI: {result['overall_aqi']} ({result['category']}) | Dominant Pollutant: {result['dominant_pollutant']} | PM2.5: {result['predicted_pm25']} µg/m³"
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            result=result,
            metadata=metadata
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error processing prediction: {str(e)}",
            metadata=metadata
        )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Check if request content type is JSON
        if request.is_json:
            data = request.json
        else:
            data = request.form
            
        model_name = data.get('model_name', 'random_forest')
        result = perform_prediction(data, model_name)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/aqi-by-coords', methods=['GET', 'POST'])
def aqi_by_coords():
    try:
        if request.method == 'POST':
            if request.is_json:
                data = request.json
            else:
                data = request.form
        else:
            data = request.args
            
        lat = data.get('latitude') or data.get('lat')
        lon = data.get('longitude') or data.get('lon')
        
        if not lat or not lon:
            return jsonify({"success": False, "error": "Latitude and Longitude are required"}), 400
            
        # Fetch weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        req = urllib.request.Request(weather_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            weather_data = json.loads(resp.read().decode())
            
        # Fetch air quality data
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        req = urllib.request.Request(aqi_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            aqi_data = json.loads(resp.read().decode())
            
        if not weather_data or 'current' not in weather_data or not aqi_data or 'current' not in aqi_data:
            return jsonify({"success": False, "error": "Failed to fetch data from API"}), 500
            
        cur_weather = weather_data['current']
        cur_aqi = aqi_data['current']
        
        pm10 = float(cur_aqi.get('pm10') or 40)
        no2 = float((cur_aqi.get('nitrogen_dioxide') or 25) * 0.53)
        so2 = float((cur_aqi.get('sulphur_dioxide') or 5) * 0.38)
        co = float((cur_aqi.get('carbon_monoxide') or 300) * 0.00087)
        o3 = float((cur_aqi.get('ozone') or 50) * 0.51)
        temp = float(cur_weather.get('temperature_2m') or 25)
        humidity = float(cur_weather.get('relative_humidity_2m') or 60)
        wind_speed = float(cur_weather.get('wind_speed_10m') or 3.0)
        
        api_pm25 = float(cur_aqi.get('pm2_5') or 15)
        
        features_dict = {
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed
        }
        
        model_name = data.get('model_name', 'random_forest')
        result = perform_prediction(features_dict, model_name)
        
        # Calculate overall AQI combining fetched PM2.5 and calculated sub-indices
        real_pm25_aqi = calculate_aqi_subindex(api_pm25, "PM2.5")
        real_sub_indices = result['sub_indices'].copy()
        real_sub_indices['PM2.5'] = real_pm25_aqi
        real_overall_aqi = max(real_sub_indices.values())
        real_category, real_color, real_desc, real_advice = get_aqi_category(real_overall_aqi)
        
        real_dominant = max(real_sub_indices, key=real_sub_indices.get)
        if real_overall_aqi == 0:
            real_dominant = "None"
            
        map_result = {
            "latitude": float(lat),
            "longitude": float(lon),
            "api_pm25": api_pm25,
            "predicted_pm25": result['predicted_pm25'],
            "overall_aqi": real_overall_aqi,
            "sub_indices": real_sub_indices,
            "dominant_pollutant": real_dominant,
            "category": real_category,
            "color": real_color,
            "description": real_desc,
            "advice": real_advice,
            "weather": {
                "temperature": temp,
                "humidity": humidity,
                "wind_speed": wind_speed
            }
        }
        
        return jsonify({"success": True, "data": map_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)

@app.route('/api/city-search', methods=['GET'])
def city_search():
    try:
        q = request.args.get('q', '').strip()
        if not q:
            return jsonify({"success": False, "error": "Query required"}), 400
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.request.quote(q)}&count=6&language=en&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        results = []
        for r in (data.get('results') or []):
            results.append({"name": r.get('name',''), "country": r.get('country',''), "admin1": r.get('admin1',''), "latitude": r.get('latitude'), "longitude": r.get('longitude')})
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/aqi-history', methods=['GET'])
def aqi_history():
    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        if not lat or not lon:
            return jsonify({"success": False, "error": "lat and lon required"}), 400
        from datetime import datetime, timedelta
        end = datetime.utcnow().strftime('%Y-%m-%d')
        start = (datetime.utcnow() - timedelta(days=6)).strftime('%Y-%m-%d')
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}&timezone=auto"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        times = raw.get('hourly', {}).get('time', [])
        pm25s = raw.get('hourly', {}).get('pm2_5', [])
        daily = {}
        for t, v in zip(times, pm25s):
            day = t[:10]
            if v is not None:
                daily.setdefault(day, []).append(v)
        days, aqis = [], []
        for day in sorted(daily.keys()):
            avg_pm25 = sum(daily[day]) / len(daily[day])
            aqi_val = calculate_aqi_subindex(avg_pm25, "PM2.5")
            days.append(day)
            aqis.append(aqi_val)
        return jsonify({"success": True, "dates": days, "aqi_values": aqis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/aqi-forecast', methods=['GET'])
def aqi_forecast():
    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        if not lat or not lon:
            return jsonify({"success": False, "error": "lat and lon required"}), 400
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&forecast_days=4&timezone=auto"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        times = raw.get('hourly', {}).get('time', [])
        pm25s = raw.get('hourly', {}).get('pm2_5', [])
        from datetime import datetime
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily = {}
        for t, v in zip(times, pm25s):
            day = t[:10]
            if day > today and v is not None:
                daily.setdefault(day, []).append(v)
        days, aqis, cats, colors = [], [], [], []
        for day in sorted(daily.keys())[:3]:
            avg_pm25 = sum(daily[day]) / len(daily[day])
            aqi_val = calculate_aqi_subindex(avg_pm25, "PM2.5")
            cat, color, _, _ = get_aqi_category(aqi_val)
            days.append(day)
            aqis.append(aqi_val)
            cats.append(cat)
            colors.append(color)
        return jsonify({"success": True, "dates": days, "aqi_values": aqis, "categories": cats, "colors": colors})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/compare', methods=['GET'])
def compare():
    try:
        lat1 = request.args.get('lat1'); lon1 = request.args.get('lon1')
        lat2 = request.args.get('lat2'); lon2 = request.args.get('lon2')
        name1 = request.args.get('name1', 'Location A')
        name2 = request.args.get('name2', 'Location B')
        if not all([lat1, lon1, lat2, lon2]):
            return jsonify({"success": False, "error": "lat1,lon1,lat2,lon2 required"}), 400
        def fetch_loc(lat, lon, name):
            aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
            req2 = urllib.request.Request(aqi_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req2, timeout=8) as r:
                d = json.loads(r.read().decode())
            cur = d.get('current', {})
            pm25 = float(cur.get('pm2_5') or 15)
            pm10 = float(cur.get('pm10') or 40)
            no2 = float((cur.get('nitrogen_dioxide') or 25) * 0.53)
            so2 = float((cur.get('sulphur_dioxide') or 5) * 0.38)
            co = float((cur.get('carbon_monoxide') or 300) * 0.00087)
            o3 = float((cur.get('ozone') or 50) * 0.51)
            sub = {"PM2.5": calculate_aqi_subindex(pm25,"PM2.5"), "PM10": calculate_aqi_subindex(pm10,"PM10"), "NO2": calculate_aqi_subindex(no2,"NO2"), "SO2": calculate_aqi_subindex(so2,"SO2"), "CO": calculate_aqi_subindex(co,"CO"), "O3": calculate_aqi_subindex(o3,"O3")}
            overall = max(sub.values())
            cat, color, desc, advice = get_aqi_category(overall)
            return {"name": name, "overall_aqi": overall, "category": cat, "color": color, "sub_indices": sub, "pm25": pm25}
        loc1 = fetch_loc(lat1, lon1, name1)
        loc2 = fetch_loc(lat2, lon2, name2)
        return jsonify({"success": True, "location1": loc1, "location2": loc2})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
