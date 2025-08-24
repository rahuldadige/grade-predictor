import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
model = joblib.load("grade_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
support_encoder = joblib.load("support_encoder.pkl")

# Title
st.title("📊 Grade Predictor App")

# Input form
st.header("Enter Student Information")

name = st.text_input("Name")
gender = st.selectbox("Gender", ["Male", "Female"])
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
study_hours = st.slider("Study Hours Per Week", 0, 40, 15)
previous_grade = st.slider("Previous Grade", 0, 100, 75)
activities = st.slider("Extracurricular Activities (Count)", 0, 10, 2)
parental_support = st.selectbox("Parental Support Level", ["Low", "Medium", "High"])

if st.button("Predict Final Grade"):
    # Preprocess inputs
    gender_encoded = gender_encoder.transform([gender])[0]
    support_encoded = support_encoder.transform([parental_support])[0]

    features = [[
        gender_encoded,
        attendance_rate,
        study_hours,
        previous_grade,
        activities,
        support_encoded
    ]]

    # Predict
    prediction = model.predict(features)[0]
    st.success(f"🎯 Predicted Final Grade for {name}: **{round(prediction, 2)}**")
