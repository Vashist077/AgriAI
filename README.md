# 🌱 SmartEco AgriCare AI System

An eco-friendly, AI-powered agricultural dashboard designed for modern sustainable farming. Built as a Final Year Engineering Project using Python, Streamlit, and Machine Learning.

## 🚀 Project Overview
SmartEco AgriCare equips farmers with data-driven, environmentally friendly solutions without requiring heavy computation. It unites three major agricultural disciplines into one centralized interface:
* **🌾 Crop Disease Detection**: Upload a leaf image. Uses a lightweight MobileNetV2 architecture tuned on the PlantVillage dataset to instantly classify diseases and recommend organic cures.
* **💧 Water Requirement Predictor**: Uses synthetic agronomical data to power a Random Forest Regressor that predicts daily Liters/Day irrigation volume dynamically against live environmental conditions, reducing general water waste.
* **🐛 Pest Outbreak Risk Evalutaion**: Utilizes Random Forest classifiers against meteorological trends to assess vulnerability vectors, prescribing bio-insecticides strictly over harmful chemical sprays.

## 🗂️ Module Architecture
* `app.py`: The main Streamlit dashboard. Unifies all ML inputs/outputs via `@st.cache_resource` for instant execution.
* `modules/crop_disease.py`: CNN architecture handling image pre-processing and model inference.
* `modules/water_prediction.py`: Tabular regression model focusing on temperature, humidity, and soil moisture scaling.
* `modules/pest_risk.py`: Categorical vulnerability assessment mapping climatic constraints to risk probability algorithms.

## 💾 Dataset Sources
1. **PlantVillage Dataset**: [Kaggle - EmmaRex/PlantDisease](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. **Crop Water Requirement**: Synthesized procedurally using `numpy` based on agronomical thresholds matching real-world N-P-K datasets. 
3. **Pest Risk Classification**: Synthesized procedurally emphasizing the positive correlation between high temperatures/humidity and infestation probability.

## ⚙️ Setup Instructions & Execution
### 1. Requirements
* Python 3.9 - 3.12 
* Pip & Virtual environment capabilities.

### 2. Installation
Ensure you clone the repository. Then, install the highly pinned requirements:
```bash
pip install -r requirements.txt
```

### 3. Execution
Launch the farmer dashboard perfectly on any device:
```bash
streamlit run app.py
```

## 📸 Dashboard Preview
*(Insert screenshots of your executing Streamlit application here before your Viva submission.)*
- `[Screenshot of Tab 1: Crop Health Image Uploader]`
- `[Screenshot of Tab 2: Water Regressor Output]`
- `[Screenshot of Tab 3: Pest Evaluation Profile]`
- `[Screenshot of Tab 4: Executive Summary]`
