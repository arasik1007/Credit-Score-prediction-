import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import joblib

df = pd.read_csv(r"C:\Users\Lalith\Desktop\vscode_project\cerdit_score\data.csv") 

print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns:\n", df.columns)

df_encoded = df.copy()

label_encoders = {}

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

print("\nCategorical columns encoded successfully")

X = df_encoded.drop("Creditworthiness", axis=1)
y = df_encoded["Creditworthiness"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "creditworthiness_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel and encoders saved successfully")
