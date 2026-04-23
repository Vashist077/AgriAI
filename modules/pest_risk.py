import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Ensure directories exist
os.makedirs('models', exist_ok=True)

# Define constants
CROP_TYPES = ['Wheat', 'Rice', 'Tomato', 'Potato', 'Maize']
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
RISK_LEVELS = ['Low', 'Medium', 'High']

# Eco-Friendly Recommendation Rules (Hardcoded)
ECO_RECOMMENDATIONS = {
    'Low': 'No action needed. Monitor weekly.',
    'Medium': 'Apply neem oil spray (5ml/litre). Introduce ladybugs for aphid control.',
    'High': 'Use Trichoderma bio-fungicide. Set pheromone traps. Avoid chemical pesticides.'
}

def generate_synthetic_data(num_samples=600):
    """
    Generates a synthetic agronomical dataset focused on pest risk conditions.
    """
    np.random.seed(42)
    
    # Generate random features
    temperatures = np.random.uniform(10, 45, num_samples) # Celsius
    humidities = np.random.uniform(20, 95, num_samples)   # Percentage
    rainfalls = np.random.uniform(0, 300, num_samples)    # mm
    wind_speeds = np.random.uniform(0, 50, num_samples)   # km/h
    crops = np.random.choice(CROP_TYPES, num_samples)
    seasons = np.random.choice(SEASONS, num_samples)

    # Establish realistic pest risk correlations
    # High temp + high humidity + low wind usually = High risk
    risk_scores = np.zeros(num_samples)
    
    for i in range(num_samples):
        score = 0
        
        # Temp factor (Pests love 25-35C)
        if 25 <= temperatures[i] <= 35:
            score += 3
        elif temperatures[i] > 35:
            score += 1
            
        # Humidity factor (Pests thrive in >70%)
        if humidities[i] > 70:
            score += 3
        elif humidities[i] > 50:
            score += 1
            
        # Wind factor (High winds disrupt flying pests)
        if wind_speeds[i] < 15:
            score += 2
        
        # Rainfall (Stagnant water/moderate rain breeds pests)
        if 50 <= rainfalls[i] <= 150:
            score += 2
            
        # Target Risk Mapping
        if score >= 7:
            risk_scores[i] = 2  # High
        elif 4 <= score < 7:
            risk_scores[i] = 1  # Medium
        else:
            risk_scores[i] = 0  # Low

    df = pd.DataFrame({
        'temperature': temperatures,
        'humidity': humidities,
        'rainfall': rainfalls,
        'wind_speed': wind_speeds,
        'crop_type': crops,
        'season': seasons,
        'risk_level': [RISK_LEVELS[int(s)] for s in risk_scores]
    })
    
    return df

def preprocess_and_train():
    """
    Loads datast, encodes features, and trains the Random Forest Classifier.
    """
    print("Generating Synthetic Pest Risk Dataset...")
    df = generate_synthetic_data()
    
    print("Encoding Categorical Variables...")
    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    le_risk = LabelEncoder()  # Ensures Low/Medium/High is safely encoded
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['season_encoded'] = le_season.fit_transform(df['season'])
    df['risk_encoded'] = le_risk.fit_transform(df['risk_level'])
    
    # Save encoders for front-end integration later
    joblib.dump(le_crop, 'models/pest_crop_encoder.pkl')
    joblib.dump(le_season, 'models/pest_season_encoder.pkl')
    joblib.dump(le_risk, 'models/pest_risk_encoder.pkl')
    
    features = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'crop_encoded', 'season_encoded']
    X = df[features]
    y = df['risk_encoded']
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier (No GPU needed)...")
    # Using Random Forest as it is extremely lightweight but highly accurate for tabular data
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Get actual string names for classification report
    target_names = le_risk.inverse_transform(np.unique(y_test))
    
    print(f"\nModel Accuracy: {accuracy*100:.2f}%")
    print(f"Weighted F1-Score: {f1:.4f}\n")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the model
    model_path = 'models/pest_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")
    
    return model

def predict_pest_risk(temp, humidity, rainfall, wind_speed, crop_type, season,
                      model_path='models/pest_model.pkl'):
    """
    Predicts pest outbreak risk and returns an eco-friendly recommendation.
    """
    if not os.path.exists(model_path):
        return {"error": "Model not trained yet."}
        
    model = joblib.load(model_path)
    le_crop = joblib.load('models/pest_crop_encoder.pkl')
    le_season = joblib.load('models/pest_season_encoder.pkl')
    le_risk = joblib.load('models/pest_risk_encoder.pkl')
    
    # Handle unseen labels cleanly
    try:
        crop_enc = le_crop.transform([crop_type])[0]
    except ValueError:
        crop_enc = 0
        
    try:
        season_enc = le_season.transform([season])[0]
    except ValueError:
        season_enc = 0
        
    # Create input vector
    features = np.array([[temp, humidity, rainfall, wind_speed, crop_enc, season_enc]])
    
    # Predict
    pred_idx = model.predict(features)[0]
    risk_level = le_risk.inverse_transform([pred_idx])[0]
    
    return {
        "risk_level": risk_level,
        "recommendation": ECO_RECOMMENDATIONS[risk_level]
    }

if __name__ == "__main__":
    print("--- Starting Pest Risk Prediction Pipeline ---")
    preprocess_and_train()
    
    print("\n--- Testing Single Inference ---")
    # Vulnerable Example Scenario: 32C, 85% humidity, 100mm rain, 5km/h wind
    test_result = predict_pest_risk(32.0, 85.0, 100.0, 5.0, 'Tomato', 'Summer')
    
    print(f"Test Input: 32°C, 85% Humidity, 100mm Rain, 5km/h Wind, Tomato, Summer")
    print(f"Predicted Risk Level: {test_result['risk_level']}")
    print(f"Eco Prescription: {test_result['recommendation']}")
