import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# 1. Load dataset
df = pd.read_csv("data/hour.csv")

# 2. Prepare features and target
# Exclude cnt, casual, registered, and instant to avoid leakage
features = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# 3. Split data into training and testing sets
# Preserve chronological order to better simulate future predictions
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Mean Absolute Error (MAE): {mae:.2f} rentals")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} rentals")
print(f"R² Score: {r2:.3f}")

# 7. Save the trained model
joblib.dump({"model": model, "features": features}, "demand_model.pkl")
print("\nModel saved as demand_model.pkl")