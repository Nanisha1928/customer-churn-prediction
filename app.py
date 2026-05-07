import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")

st.title("📈 Customer Churn Prediction")

st.write("Enter customer details below:")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
monthly = st.number_input("Monthly Charges", 0.0, 500.0)
tenure = st.slider("Tenure (months)", 0, 72)

# Prediction button
if st.button("Predict"):

    # Simple sample input
    data = {
        "SeniorCitizen": [senior],
        "MonthlyCharges": [monthly],
        "tenure": [tenure],
        "gender_Male": [1 if gender == "Male" else 0]
    }

    input_df = pd.DataFrame(data)

    # Add missing columns if required
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns correctly
    input_df = input_df[model_features]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")