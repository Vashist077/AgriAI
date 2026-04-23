import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import tempfile

# Attempt to load TF, handles error if user hasn't fixed the Long Path issue yet
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Make sure directories exist for temp files
os.makedirs('assets', exist_ok=True)

# -----------------------------------------
# CONSTANTS & DEFINITIONS
# -----------------------------------------
CROP_TYPES = ['Wheat', 'Rice', 'Tomato', 'Potato', 'Maize']
GROWTH_STAGES = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']

DISEASE_CLASSES = ['Diseased', 'Healthy', 'Nutrient_Deficiency']
DISEASE_EXPLANATIONS = {
    'Healthy': 'Great news! Your crop is healthy. Continue standard organic maintenance.',
    'Diseased': 'We detected a pathogen. Isolate affected leaves and apply neem oil or bio-fungicide immediately.',
    'Nutrient_Deficiency': 'Nutrient imbalance detected. Verify soil N-P-K levels and apply organic compost or vermicompost.'
}

ECO_TIPS_WATER = [
    "Use drip irrigation to save up to 40% of water.",
    "Apply organic mulch to retain soil moisture.",
    "Water crops early morning to reduce evaporation loss.",
    "Consider rainwater harvesting systems to supplement irrigation.",
    "Use precision agriculture techniques to water only when soil moisture is low."
]

ECO_RECOMMENDATIONS_PEST = {
    'Low': 'No immediate action needed. Continue weekly visual monitoring.',
    'Medium': 'Apply neem oil spray (5ml/litre) specifically to undersides of leaves. Introduce ladybugs for aphid control.',
    'High': 'Critical risk! Use Trichoderma bio-fungicide. Set pheromone traps immediately. Avoid broad-spectrum chemical pesticides to protect local bees.'
}

# -----------------------------------------
# STREAMLIT CACHED MODEL LOADING
# -----------------------------------------
@st.cache_resource
def load_models():
    models = {}
    
    # 1. Load Crop Disease Model (TF)
    if TF_AVAILABLE:
        try:
            models['disease'] = tf.keras.models.load_model('models/crop_disease_model.h5')
        except Exception as e:
            models['disease'] = None
            
    # 2. Load Water Prediction Model & Encoders
    try:
        models['water'] = joblib.load('models/water_model.pkl')
        models['water_crop_enc'] = joblib.load('models/crop_encoder.pkl')
        models['water_stage_enc'] = joblib.load('models/stage_encoder.pkl')
    except Exception:
        models['water'] = None
        
    # 3. Load Pest Risk Model & Encoders
    try:
        models['pest'] = joblib.load('models/pest_model.pkl')
        models['pest_crop_enc'] = joblib.load('models/pest_crop_encoder.pkl')
        models['pest_season_enc'] = joblib.load('models/pest_season_encoder.pkl')
        models['pest_risk_enc'] = joblib.load('models/pest_risk_encoder.pkl')
    except Exception:
        models['pest'] = None
        
    return models

# -----------------------------------------
# INFERENCE LOGIC (Reimplemented for Streamlit)
# -----------------------------------------
def predict_disease_ui(image_file, model):
    if not TF_AVAILABLE or model is None:
        return None, None, "TensorFlow environment not fully configured."
        
    # Save uploaded file temporarily for Keras load_img
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = tmp_file.name

    img = load_img(tmp_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]
    os.remove(tmp_path) # Clean up
    
    class_idx = np.argmax(predictions)
    class_name = sorted(DISEASE_CLASSES)[class_idx]
    confidence = float(predictions[class_idx]) * 100
    
    return class_name, confidence, DISEASE_EXPLANATIONS.get(class_name, "No explanation available.")

def predict_water_ui(temp, humidity, soil_moisture, crop_type, growth_stage, models):
    if models.get('water') is None:
        return None, "Water Prediction Model not found. Train it first."
        
    try:
        crop_enc = models['water_crop_enc'].transform([crop_type])[0]
    except ValueError:
        crop_enc = 0
    try:
        stage_enc = models['water_stage_enc'].transform([growth_stage])[0]
    except ValueError:
        stage_enc = 0
        
    features = np.array([[temp, humidity, soil_moisture, crop_enc, stage_enc]])
    water_req = models['water'].predict(features)[0]
    
    tip = np.random.choice(ECO_TIPS_WATER)
    return round(float(water_req), 1), tip

def predict_pest_ui(temp, humidity, rainfall, wind_speed, crop_type, season, models):
    if models.get('pest') is None:
        return None, "Pest Prediction Model not found. Train it first."
        
    try:
        crop_enc = models['pest_crop_enc'].transform([crop_type])[0]
    except ValueError:
        crop_enc = 0
    try:
        season_enc = models['pest_season_enc'].transform([season])[0]
    except ValueError:
        season_enc = 0
        
    features = np.array([[temp, humidity, rainfall, wind_speed, crop_enc, season_enc]])
    pred_idx = models['pest'].predict(features)[0]
    risk_level = models['pest_risk_enc'].inverse_transform([pred_idx])[0]
    
    return risk_level, ECO_RECOMMENDATIONS_PEST.get(risk_level, "")

# -----------------------------------------
# UI CONFIGURATION & LAYOUT
# -----------------------------------------
st.set_page_config(page_title="SmartEco AgriCare", page_icon="🌿", layout="wide")

st.title("🌿 SmartEco AgriCare AI System")
st.markdown("Your trusted, eco-friendly AI advisor for sustainable farming.")

# Run model loading
models_cache = load_models()

# Sidebar Inputs
st.sidebar.header("🚜 Farm Configuration")
selected_crop = st.sidebar.selectbox("Current Crop Type", CROP_TYPES)
selected_stage = st.sidebar.selectbox("Growth Stage", GROWTH_STAGES)
st.sidebar.markdown("---")
st.sidebar.info("Data entered here will be synced across all analytical tabs.")

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌱 Crop Health", "💧 Water Advisor", "🐛 Pest Alert", "📊 Summary Dashboard"])

# state dictionaries to pass data to summary tab
if 'disease_result' not in st.session_state: st.session_state.disease_result = None
if 'water_result' not in st.session_state: st.session_state.water_result = None
if 'pest_result' not in st.session_state: st.session_state.pest_result = None

with tab1:
    st.header("Upload Leaf Image for Disease Detection")
    uploaded_file = st.file_uploader("Upload a clear picture of the affected leaf (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        colA, colB = st.columns([1, 2])
        with colA:
            st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)
            
        with colB:
            st.write("### AI Analysis Results")
            with st.spinner("Analyzing cell structures via MobileNetV2..."):
                d_class, d_conf, d_exp = predict_disease_ui(uploaded_file, models_cache.get('disease'))
                
            if d_class is None:
                st.error("⚠️ AI Module offline. Please enable TensorFlow Long Paths to use this feature.")
            else:
                st.session_state.disease_result = {"class": d_class, "conf": d_conf, "exp": d_exp}
                
                # Dynamic coloring
                if d_class == 'Healthy':
                    st.success(f"**Diagnosis:** {d_class}")
                elif d_class == 'Nutrient_Deficiency':
                    st.warning(f"**Diagnosis:** {d_class.replace('_', ' ')}")
                else:
                    st.error(f"**Diagnosis:** {d_class}")
                    
                st.progress(int(d_conf) / 100)
                st.write(f"*AI Confidence Score:* {d_conf:.1f}%")
                
                st.info(f"**Organic Recommendation:** {d_exp}")

with tab2:
    st.header("Dynamic Water Requirement Calculator")
    st.write(f"Calculating for **{selected_crop}** at **{selected_stage}** stage.")
    
    c1, c2, c3 = st.columns(3)
    t_temp = c1.slider("Temperature (°C)", min_value=10.0, max_value=50.0, value=25.0)
    t_hum = c2.slider("Air Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
    t_soil = c3.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=40.0)
    
    if st.button("Calculate Daily Water Needs", use_container_width=True):
        req, tip = predict_water_ui(t_temp, t_hum, t_soil, selected_crop, selected_stage, models_cache)
        if req is None:
            st.error(tip)
        else:
            st.session_state.water_result = {"req": req, "tip": tip}
            
            st.success(f"### Optimal Irrigation Volume: {req} Litres / Day")
            st.info(f"💡 **Eco-Tip:** {tip}")
            
            # Show the generated feature importance graph if it exists
            if os.path.exists('assets/feature_importance_water.png'):
                with st.expander("View AI Decision Drivers (Feature Importance)"):
                    st.image('assets/feature_importance_water.png')

with tab3:
    st.header("Hyper-Local Pest Outbreak Alert")
    st.write(f"Evaluating vulnerabilities for **{selected_crop}**.")
    
    col_w1, col_w2 = st.columns(2)
    p_temp = col_w1.number_input("Average Temperature (°C)", value=30.0)
    p_hum = col_w1.number_input("Average Humidity (%)", value=75.0)
    p_rain = col_w2.number_input("Last 24h Rainfall (mm)", value=0.0)
    p_wind = col_w2.number_input("Wind Speed (km/h)", value=12.0)
    
    selected_season = st.selectbox("Current Season", SEASONS)
    
    if st.button("Evaluate Pest Risk Profile", use_container_width=True):
        r_level, r_tip = predict_pest_ui(p_temp, p_hum, p_rain, p_wind, selected_crop, selected_season, models_cache)
        if r_level is None:
            st.error(r_tip)
        else:
            st.session_state.pest_result = {"level": r_level, "tip": r_tip}
            
            if r_level == 'Low':
                st.success(f"### Risk Level: LOW ❇️")
            elif r_level == 'Medium':
                st.warning(f"### Risk Level: MEDIUM 🚸")
            else:
                st.error(f"### Risk Level: HIGH 🚨")
                
            st.info(f"🛡️ **Action Plan:** {r_tip}")

with tab4:
    st.header("🧑‍🌾 Farmer's Executive Summary")
    st.write(f"Consolidated analysis for your **{selected_stage} {selected_crop}** field.")
    
    sum_c1, sum_c2, sum_c3 = st.columns(3)
    
    with sum_c1:
        st.subheader("🌱 Health Status")
        if st.session_state.disease_result:
            dr = st.session_state.disease_result
            if dr['class'] == 'Healthy':
                st.success(f"**{dr['class']}** ({dr['conf']:.1f}%)")
            else:
                st.error(f"**{dr['class']}** ({dr['conf']:.1f}%)")
        else:
            st.write("Awaiting analysis...")
            
    with sum_c2:
        st.subheader("💧 Irrigation")
        if st.session_state.water_result:
            st.info(f"**{st.session_state.water_result['req']} L/Day**")
        else:
            st.write("Awaiting calculation...")
            
    with sum_c3:
        st.subheader("🐛 Pest Risk")
        if st.session_state.pest_result:
            pr = st.session_state.pest_result
            if pr['level'] == 'Low':
                st.success("**Low Risk**")
            elif pr['level'] == 'Medium':
                st.warning("**Medium Risk**")
            else:
                st.error("**High Risk**")
        else:
            st.write("Awaiting evaluation...")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Eco-friendly farming powered by AI</p>", unsafe_allow_html=True)
