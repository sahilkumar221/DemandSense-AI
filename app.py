import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="DemandSense-AI | Demand Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .subtitle {
        color: #808A9A;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

import os
import pickle
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_data():
    return pd.read_csv("data/hour.csv")

@st.cache_resource
def load_model():
    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum",
        "windspeed"
    ]

    # Load the dataset
    df = load_data()

    # Use the saved model if it can be loaded.
    if os.path.exists("demand_model.pkl"):
        try:
            with open("demand_model.pkl", "rb") as file:
                saved = pickle.load(file)

            return saved
        except (ModuleNotFoundError, AttributeError, ImportError, pickle.UnpicklingError):
            st.warning("The saved model could not be loaded. Training a replacement.")

    # Train a model if no usable saved model is available.
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(df[features], df["cnt"])

    return {"model": model, "features": features}

df = load_data()
loaded = load_model()
model = loaded["model"]
features = loaded["features"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🚲 DemandSense-AI")
st.markdown(
    '<p class="subtitle">Hourly Bike Rental Demand '
    'Forecasting & Analytics</p>',
    unsafe_allow_html=True,
)

st.caption(
    "Machine learning powered demand estimates "
    "using historical bike-sharing data."
)

st.divider()

# --------------------------------------------------
# Historical dataset overview
# --------------------------------------------------
total_records = len(df)
average_demand = df["cnt"].mean()
peak_hour = int(
    df.groupby("hr")["cnt"].mean().idxmax()
)
total_rentals = int(df["cnt"].sum())

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Hourly Records", f"{total_records:,}")
c2.metric("Average Hourly Rentals", f"{average_demand:.1f}")
c3.metric("Peak Demand Hour", f"{peak_hour:02d}:00")
c4.metric("Total Recorded Rentals", f"{total_rentals:,}")
st.caption(
    "These figures summarize the loaded dataset; "
    "they are not live rental counts."
)

st.divider()

# --------------------------------------------------
# Prediction controls
# --------------------------------------------------
st.sidebar.title("🔧 Prediction Settings")
st.sidebar.write("Choose conditions for an hourly estimate.")

with st.sidebar.form("prediction_form"):
    season = st.selectbox(
        "Season",
        [1, 2, 3, 4],
        index=1,
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter",
        }[x],
    )

    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=5,
    )

    hour = st.slider("Hour of day", 0, 23, 8)

    year = st.selectbox(
        "Dataset year",
        [0, 1],
        index=1,
        format_func=lambda x: {
            0: "2011 (year 0)",
            1: "2012 (year 1)",
        }[x],
    )

    weekday = st.selectbox(
        "Day of week",
        list(range(7)),
        index=2,
        format_func=lambda x: {
            0: "Sunday",
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
        }[x],
    )

    workingday = st.selectbox(
        "Working day?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    holiday = st.selectbox(
        "Holiday?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )

    weather = st.selectbox(
        "Weather",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear or partly cloudy",
            2: "Mist or cloudy",
            3: "Light rain or snow",
            4: "Heavy rain or snow",
        }[x],
    )

    temp = st.slider(
        "Temperature (normalized)",
        0.0, 1.0, 0.60, 0.01,
    )

    atemp = st.slider(
        "Feels-like temperature (normalized)",
        0.0, 1.0, 0.58, 0.01,
    )

    humidity = st.slider(
        "Humidity (normalized)",
        0.0, 1.0, 0.60, 0.01,
    )

    windspeed = st.slider(
        "Wind speed (normalized)",
        0.0, 1.0, 0.20, 0.01,
    )

    predict_clicked = st.form_submit_button(
        "Predict Bike Demand",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Prediction result
# --------------------------------------------------
st.subheader("Demand Prediction")

if predict_clicked:
    sample = {
        "season": season,
        "yr": year,
        "mnth": month,
        "hr": hour,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temp,
        "atemp": atemp,
        "hum": humidity,
        "windspeed": windspeed,
    }

    input_df = pd.DataFrame([sample], columns=features)
    prediction = max(
        0,
        round(float(model.predict(input_df)[0])),
    )

    st.session_state["prediction"] = prediction
    st.session_state["prediction_inputs"] = sample

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    sample = st.session_state["prediction_inputs"]

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Predicted Hourly Rentals",
        f"{prediction:,}",
    )
    p2.metric("Selected Hour", f"{sample['hr']:02d}:00")
    p3.metric("Selected Month", sample["mnth"])

    if prediction >= 300:
        st.success(
            "High estimated demand. Consider preparing "
            "additional bikes."
        )
    elif prediction >= 100:
        st.info(
            "Moderate estimated demand. Monitor bike availability."
        )
    else:
        st.warning(
            "Low estimated demand. Review local operating needs."
        )

    st.caption(
        "This is a model estimate, not a guaranteed count. "
        "The thresholds are illustrative planning rules."
    )
else:
    st.info(
        "Adjust the inputs in the sidebar and click "
        "'Predict Bike Demand' to generate an estimate."
    )

st.divider()

# --------------------------------------------------
# Historical demand analytics
# --------------------------------------------------
st.subheader("Historical Demand Analytics")

tab1, tab2, tab3 = st.tabs(
    ["Hourly Trends", "Seasonal Analysis", "Weather Analysis"]
)

with tab1:
    hourly = (
        df.groupby("hr", as_index=False)["cnt"]
        .mean()
        .rename(columns={"cnt": "Average rentals"})
    )

    st.markdown("**Average rentals by hour of day**")
    st.line_chart(
        hourly,
        x="hr",
        y="Average rentals",
    )

    st.caption(
        "Use this chart to identify busy hours and quieter periods."
    )

with tab2:
    season_names = {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter",
    }

    seasonal = (
        df.groupby("season", as_index=False)["cnt"]
        .mean()
    )
    seasonal["Season"] = seasonal["season"].map(season_names)
    seasonal = seasonal.rename(
        columns={"cnt": "Average rentals"}
    )

    st.markdown("**Average rentals by season**")
    st.bar_chart(
        seasonal,
        x="Season",
        y="Average rentals",
    )

with tab3:
    weather_names = {
        1: "Clear",
        2: "Mist / cloudy",
        3: "Light precipitation",
        4: "Heavy precipitation",
    }

    weather_data = (
        df.groupby("weathersit", as_index=False)["cnt"]
        .mean()
    )
    weather_data["Weather"] = weather_data[
        "weathersit"
    ].map(weather_names)
    weather_data = weather_data.rename(
        columns={"cnt": "Average rentals"}
    )

    st.markdown("**Average rentals by weather condition**")
    st.bar_chart(
        weather_data,
        x="Weather",
        y="Average rentals",
    )

st.divider()

# --------------------------------------------------
# Model explainability
# --------------------------------------------------
st.subheader("What Influences the Model?")

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

st.markdown(
    "Feature importance indicates how much each input contributes "
    "to the Random Forest's decisions. It does not prove causation."
)

st.bar_chart(
    importance.head(10).sort_values("Importance"),
    x="Importance",
    y="Feature",
    horizontal=True,
)

with st.expander("View feature importance values"):
    st.dataframe(
        importance.reset_index(drop=True),
        use_container_width=True,
    )

st.divider()

st.caption(
    "DemandSense-AI | Built with Python, Pandas, "
    "Scikit-learn and Streamlit"
)

