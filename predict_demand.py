import pandas as pd
import joblib

# Load the saved model
saved = joblib.load("demand_model.pkl")
model = saved["model"]
features = saved["features"]

# Example conditions for one hour
sample = {
    "season": 2,
    "yr": 1,
    "mnth": 6,
    "hr": 8,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.60,
    "atemp": 0.58,
    "hum": 0.60,
    "windspeed": 0.20,
}

# Arrange inputs in the exact order used during training
input_df = pd.DataFrame([sample], columns=features)

# Predict hourly rentals
prediction = model.predict(input_df)[0]

print("DemandSense-AI Prediction")
print("------------------------")
print("Hour:", sample["hr"])
print("Month:", sample["mnth"])
print(f"Predicted bike rentals: {max(0, round(prediction))}")

