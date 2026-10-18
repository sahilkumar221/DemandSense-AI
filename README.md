
# 🚲 DemandSense-AI

An interactive bike rental demand prediction dashboard built using Python and Machine Learning.

## Features
- Explore historical bike rental data
- Visualize demand by hour, season, and weather
- Predict bike rental demand using a Random Forest model
- View feature importance
- Evaluate model performance against a baseline

## Model Performance

Evaluated on the final 20% of the dataset in chronological order.

| Metric | Random Forest | Mean Baseline |
|---|---:|---:|
| MAE | 44.99 | 174.98 |
| RMSE | 69.56 | 232.61 |
| R² | 0.900 | -0.113 |

These results use a holdout period from the available dataset, not an independent external test.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Run Locally

1. Install Python 3.13.
2. Open a terminal in the project folder.
3. Install the required packages:

   `python -m pip install -r requirements.txt`

4. Start the dashboard:

   `python -m streamlit run app.py`

## Project Files
- `app.py` — interactive dashboard
- `explore_data.py` — data exploration
- `visualize_data.py` — charts
- `train_model.py` — model training
- `predict_demand.py` — sample prediction
- `evaluate_model.py` — model evaluation
- `data/hour.csv` — hourly bike rental dataset

## Limitations
Predictions are estimates based on historical data from 2011–2012 and may not represent current demand.

## Dataset Attribution

This project uses the Bike Sharing Dataset by Hadi Fanaee-T, obtained from the UCI Machine Learning Repository.

Source: https://doi.org/10.24432/C5W894

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset contains hourly bike rental records from 2011–2012.

