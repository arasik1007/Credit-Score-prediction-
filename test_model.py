import joblib
import pandas as pd

model = joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\creditworthiness_model.pkl")
label_encoders = joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\label_encoders.pkl")

print("Model and encoders loaded successfully")

input_data = {
    "Age": 30,
    "Gender": "Male",
    "Education": "Bachelor",
    "Income": 50000,
    "Debt": 12000,
    "Credit_Score": 720,
    "Loan_Amount": 200000,
    "Loan_Term": 36,
    "Num_Credit_Cards": 2,
    "Payment_History": "Good",
    "Employment_Status": "Employed",
    "Residence_Type": "Owned",
    "Marital_Status": "Single"
}

input_df = pd.DataFrame([input_data])

for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

result_map = {
    0: "Bad",
    1: "Good"
}

print("\nPrediction Result")
print("------------------")
print("Creditworthiness:", result_map[int(prediction[0])])
print("Confidence:", round(max(prediction_proba[0]) * 100, 2), "%")
