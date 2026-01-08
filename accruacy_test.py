import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\data.csv") 

model = joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\creditworthiness_model.pkl")
encoders =joblib.load(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\label_encoders.pkl")

df_encoded = df.copy()
for col in df_encoded.columns:
    if col in encoders:
        df_encoded[col] = encoders[col].transform(df_encoded[col])

X = df_encoded.drop("Creditworthiness", axis=1)
y = df_encoded["Creditworthiness"]

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print(f"Model Accuracy on full dataset: {accuracy * 100:.2f}%")
