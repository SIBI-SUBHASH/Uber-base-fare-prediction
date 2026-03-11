import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Uber Fare Prediction AI", layout="centered")

st.title("🚕 Uber Fare Prediction")
st.write("Predict the estimated fare of an Uber ride using Machine Learning.")

# -----------------------------------
# Load Trained Model (with XGBoost version fix)
# -----------------------------------
@st.cache_resource
def load_model():
    model_path = "uber_fare_model.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {os.path.abspath(model_path)}")
        return None

    # --- Strategy 1: Standard joblib load ---
    try:
        model = joblib.load(model_path)
        # Verify the model is actually fitted by checking internal state
        if hasattr(model, 'get_booster'):
            _ = model.get_booster()  # This will raise if not fitted
        return model
    except Exception as e1:
        st.warning(f"joblib load failed ({e1}), trying alternative loading...")

    # --- Strategy 2: Pickle load ---
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if hasattr(model, 'get_booster'):
            _ = model.get_booster()
        return model
    except Exception as e2:
        st.warning(f"pickle load also failed ({e2}), trying XGBoost native format...")

    # --- Strategy 3: Save/reload via XGBoost native format ---
    # This fixes version mismatch by re-serializing through XGBoost's own I/O
    try:
        import xgboost as xgb

        # Load raw object ignoring booster state
        with open(model_path, "rb") as f:
            raw_model = pickle.load(f)

        # Extract the booster JSON if available and rebuild
        temp_path = "temp_booster.json"
        raw_model.save_model(temp_path)

        new_model = xgb.XGBRegressor()
        new_model.load_model(temp_path)
        os.remove(temp_path)

        st.success("✅ Model loaded via XGBoost native format (version compatibility fix applied).")
        return new_model

    except Exception as e3:
        st.error(
            f"All model loading strategies failed.\n\n"
            f"**Root Cause:** Your `uber_fare_model.pkl` was saved with a different version of XGBoost "
            f"than the one currently installed. This causes a `NotFittedError` even though the model was trained.\n\n"
            f"**Fix:** Re-run your training notebook (`Uber_Fare_Prediction__Regression_.ipynb`) "
            f"with your current environment to regenerate `uber_fare_model.pkl`, then restart the app.\n\n"
            f"Error details: {e3}"
        )
        return None

model = load_model()

# -----------------------------------
# User Inputs
# -----------------------------------
st.subheader("Enter Ride Details")

col1, col2 = st.columns(2)

with col1:
    pickup_longitude = st.number_input("Pickup Longitude", value=-73.985428, format="%.6f")
    pickup_latitude = st.number_input("Pickup Latitude", value=40.748817, format="%.6f")

with col2:
    dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.984427, format="%.6f")
    dropoff_latitude = st.number_input("Dropoff Latitude", value=40.758817, format="%.6f")

passenger_count = st.slider("Passenger Count", 1, 6, 1)

col3, col4 = st.columns(2)
with col3:
    hour = st.slider("Hour of Day", 0, 23, 12)
    day = st.slider("Day of Month", 1, 31, 15)
with col4:
    month = st.slider("Month", 1, 12, 6)
    weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 3)

# -----------------------------------
# Calculate Distance (Haversine)
# -----------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

distance = haversine(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
distance = max(distance, 0)

st.info(f"📍 Estimated Distance: **{distance:.2f} km**")

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🚕 Predict Fare", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please check the error messages above.")
    else:
        input_data = pd.DataFrame(
            [[
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                passenger_count,
                hour,
                day,
                month,
                weekday,
                distance,
            ]],
            columns=[
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
                "hour",
                "day",
                "month",
                "weekday",
                "distance_km",
            ],
        )

        try:
            prediction = model.predict(input_data)[0]
            prediction = max(prediction, 2.50)  # Uber minimum fare guard
            st.success(f"💰 Estimated Uber Fare: **${prediction:.2f}**")
        except Exception as e:
            st.error(
                f"Prediction failed: {e}\n\n"
                "This is likely an XGBoost version mismatch. "
                "Please re-train and re-save the model in your current environment."
            )

# -----------------------------------
# Feature Importance
# -----------------------------------
try:
    if model is not None and hasattr(model, "feature_importances_"):
        st.subheader("📊 Model Feature Importance")

        features = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "hour",
            "day",
            "month",
            "weekday",
            "distance_km",
        ]

        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [features[i] for i in sorted_idx],
            importance[sorted_idx],
            color="steelblue",
        )
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
except Exception:
    pass
