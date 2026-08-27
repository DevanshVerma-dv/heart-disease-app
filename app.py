import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("columns.pkl")

st.title("Heart Disease Prediction App")
st.markdown("Provide the following information: ")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", ["Greater than 120 mg/dl", "Less than 120 mg/dl"])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    raw_inp = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': chol,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex': 1 if sex == "Male" else 0,
        'ChestPainType': chest_pain,
        'FastingBS': 1 if fasting_bs == "Greater than 120 mg/dl" else 0,
        'RestingECG': resting_ecg,
        'ExerciseAngina': 1 if exercise_angina == "Yes" else 0,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_inp])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]

    if (prediction == 1):
        st.error("⚠️The model predicts that you may have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("✅The model predicts that you are unlikely to have heart disease. However, please consult a healthcare professional for a comprehensive assessment.")