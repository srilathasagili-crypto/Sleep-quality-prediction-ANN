import streamlit as st
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Sleep Quality Prediction",
    page_icon="😴",
    layout="centered"
)

st.title("😴 Sleep Quality Prediction using ANN")
st.write("Enter the details below to predict Sleep Quality.")

# ==========================================
# Load Model
# ==========================================

model = load_model("sleep_quality_ann.keras")

# ==========================================
# Load Artifacts
# ==========================================

with open("artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

scaler = artifacts["scaler"]

gender_encoder = artifacts["gender_encoder"]
occupation_encoder = artifacts["occupation_encoder"]
bmi_encoder = artifacts["bmi_encoder"]
sleep_disorder_encoder = artifacts["sleep_disorder_encoder"]

feature_columns = artifacts["feature_columns"]

# ==========================================
# User Inputs
# ==========================================

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

occupation = st.selectbox(
    "Occupation",
    occupation_encoder.classes_
)

sleep_duration = st.number_input(
    "Sleep Duration (Hours)",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

physical_activity = st.number_input(
    "Physical Activity Level",
    min_value=0,
    max_value=100,
    value=50
)

stress_level = st.number_input(
    "Stress Level",
    min_value=0,
    max_value=10,
    value=5
)

bmi = st.selectbox(
    "BMI Category",
    bmi_encoder.classes_
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=200,
    value=70
)

daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    max_value=50000,
    value=8000
)

sleep_disorder = st.selectbox(
    "Sleep Disorder",
    sleep_disorder_encoder.classes_
)

systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=40,
    max_value=150,
    value=80
)

# ==========================================
# Prediction
# ==========================================

if st.button("Predict Sleep Quality"):

    gender = gender_encoder.transform([gender])[0]
    occupation = occupation_encoder.transform([occupation])[0]
    bmi = bmi_encoder.transform([bmi])[0]
    sleep_disorder = sleep_disorder_encoder.transform([sleep_disorder])[0]

    input_data = pd.DataFrame([[
        gender,
        age,
        occupation,
        sleep_duration,
        physical_activity,
        stress_level,
        bmi,
        heart_rate,
        daily_steps,
        sleep_disorder,
        systolic_bp,
        diastolic_bp
    ]], columns=feature_columns)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    sleep_quality = float(prediction[0][0])

    sleep_quality = max(0, min(10, sleep_quality))

    st.success(f"Predicted Sleep Quality: {sleep_quality:.2f} / 10")

    if sleep_quality >= 8:
        st.success("😊 Excellent Sleep Quality")

    elif sleep_quality >= 6:
        st.info("🙂 Good Sleep Quality")

    elif sleep_quality >= 4:
        st.warning("😐 Average Sleep Quality")

    else:
        st.error("😴 Poor Sleep Quality")
