import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# Define constants
CROP_TYPES = ['Wheat', 'Rice', 'Tomato', 'Potato', 'Maize']
GROWTH_STAGES = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']

# Organic/Eco-friendly tips for water conservation
ECO_TIPS = [
    "Use drip irrigation to save up to 40% of water.",
    "Apply organic mulch to retain soil moisture.",
    "Water crops early morning to reduce evaporation loss.",
    "Consider rainwater harvesting systems to supplement irrigation.",
    "Use precision agriculture techniques to water only when soil moisture is low."
]

def generate_synthetic_data(num_samples=500):
    """
    Generates a realistic synthetic agronomical dataset.
    """
    np.random.seed(42)
    
    # Generate random features
    temperatures = np.random.uniform(15, 40, num_samples) # Celsius
    humidities = np.random.uniform(20, 90, num_samples)   # Percentage
    soil_moistures = np.random.uniform(10, 80, num_samples) # Percentage
    crops = np.random.choice(CROP_TYPES, num_samples)
    stages = np.random.choice(GROWTH_STAGES, num_samples)

    # Base rules for calculating synthetic 'liters per day'
    # Base = 10, Hotter = more water, lower humidity = more water, lower soil moisture = more water
    water_req = 10 + (temperatures * 0.5) - (humidities * 0.1) - (soil_moistures * 0.2)
    
    # Adjust based on crop type
    crop_factor = {'Rice': 15, 'Wheat': 5, 'Maize': 8, 'Tomato': 4, 'Potato': 3}
    water_req += np.array([crop_factor[c] for c in crops])
    
    # Adjust based on growth stage
    stage_factor = {'Seedling': 2, 'Vegetative': 5, 'Flowering': 8, 'Fruiting': 10, 'Mature': 3}
    water_req += np.array([stage_factor[s] for s in stages])
    
    # Add some noise for realism
    water_req += np.random.normal(0, 2, num_samples)
    
    # Ensure no negative water requirements
    water_req = np.maximum(water_req, 1.0)

    df = pd.DataFrame({
        'temperature': temperatures,
        'humidity': humidities,
        'soil_moisture': soil_moistures,
        'crop_type': crops,
        'growth_stage': stages,
        'water_liters_per_day': water_req
    })
    
    return df

def preprocess_and_train():
    """
    Loads data, encodes features, and trains the Random Forest model.
    """
    print("Generating Synthetic Dataset...")
    df = generate_synthetic_data()
    
    print("Encoding Categorical Variables...")
    le_crop = LabelEncoder()
    le_stage = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['stage_encoded'] = le_stage.fit_transform(df['growth_stage'])
    
    # Save encoders for future inference
    joblib.dump(le_crop, 'models/crop_encoder.pkl')
    joblib.dump(le_stage, 'models/stage_encoder.pkl')
    
    features = ['temperature', 'humidity', 'soil_moisture', 'crop_encoded', 'stage_encoded']
    X = df[features]
    y = df['water_liters_per_day']
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f} L/day")
    print(f"MAE: {mae:.2f} L/day")
    print(f"R² Score: {r2:.4f}")
    
    # Feature Importance Plot
    print("Generating Feature Importance Plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.title("Feature Importances for Water Prediction")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='green')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('assets/feature_importance_water.png')
    print("Saved plot to assets/feature_importance_water.png")
    
    # Save model
    model_path = 'models/water_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")
    
    return model, le_crop, le_stage

def predict_water(temp, humidity, soil_moisture, crop_type, growth_stage, 
                  model_path='models/water_model.pkl'):
    """
    Predicts required water given environmental variables.
    """
    if not os.path.exists(model_path):
        return {"error": "Model not trained yet."}
        
    model = joblib.load(model_path)
    le_crop = joblib.load('models/crop_encoder.pkl')
    le_stage = joblib.load('models/stage_encoder.pkl')
    
    # Handle unseen labels gracefully for the viva demo
    try:
        crop_enc = le_crop.transform([crop_type])[0]
    except ValueError:
        crop_enc = 0 # Default fallback
        
    try:
        stage_enc = le_stage.transform([growth_stage])[0]
    except ValueError:
        stage_enc = 0
        
    features = np.array([[temp, humidity, soil_moisture, crop_enc, stage_enc]])
    val = model.predict(features)[0]
    
    # Pick a random eco-tip
    tip = np.random.choice(ECO_TIPS)
    
    return {
        "recommended_water_liters_per_day": round(float(val), 2),
        "eco_friendly_tip": tip
    }

if __name__ == "__main__":
    print("--- Starting Water Prediction Pipeline ---")
    model, le_crop, le_stage = preprocess_and_train()
    
    print("\n--- Testing Single Inference ---")
    # Example input: 30°C, 45% moisture, 30% soil moisture, Tomato, Flowering Stage
    result = predict_water(30.0, 45.0, 30.0, 'Tomato', 'Flowering')
    print(f"Test Input: 30°C, 45% humidity, 30% soil moisture, Tomato, Flowering")
    print(f"Prediction Output:")
    print(f" - Water Required: {result['recommended_water_liters_per_day']} L/day")
    print(f" - Expert Advice: {result['eco_friendly_tip']}")
