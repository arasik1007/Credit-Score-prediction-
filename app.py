import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\creditworthiness_model.pkl")
label_encoders = joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\label_encoders.pkl")

st.title("💳 Creditworthiness Prediction System")

st.markdown(
    """
    This application predicts whether a person has **Good** or **Bad**
    creditworthiness based on financial and personal details.
    """
)

columns = [
    "Age",
    "Gender",
    "Education",
    "Income",
    "Debt",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
    "Num_Credit_Cards",
    "Payment_History",
    "Employment_Status",
    "Residence_Type",
    "Marital_Status"
]

categorical_features = label_encoders.keys()

user_input = {}

st.subheader("Enter Applicant Details")

for feature in columns:
    if feature in categorical_features:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"{feature}", options)
    else:
        user_input[feature] = st.number_input(
            f"{feature}", min_value=0.0, value=0.0
        )

if st.button("Predict Creditworthiness"):
    input_data = []

    for feature in columns:
        val = user_input[feature]
        if feature in categorical_features:
            val = label_encoders[feature].transform([val])[0]
        input_data.append(val)

    input_array = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result_map = {0: "Bad", 1: "Good"}

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Creditworthiness: GOOD")
    else:
        st.error("❌ Creditworthiness: BAD")

    st.info(f"Confidence: **{max(probability) * 100:.2f}%**")
