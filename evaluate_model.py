
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Load data and saved model
df = pd.read_csv("data/hour.csv")
saved = joblib.load("demand_model.pkl")

model = saved["model"]
features = saved["features"]

X = df[features]
y = df["cnt"]

# Use the final 20% as the test set,
# matching the original chronological split
split_index = int(len(df) * 0.8)

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Model predictions
predictions = model.predict(X_test)

# Baseline: predict the training-set average every time
y_train = y.iloc[:split_index]
baseline_predictions = np.full(
    len(y_test),
    y_train.mean(),
)

def report_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

report_metrics("Random Forest", y_test, predictions)

report_metrics(
    "Mean baseline",
    y_test,
    baseline_predictions,
)

# Save actual and predicted values for further analysis
results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
})

results["error"] = (
    results["actual"] - results["predicted"]
)

results.to_csv("prediction_results.csv", index=False)

print("\nSaved predictions to prediction_results.csv")
