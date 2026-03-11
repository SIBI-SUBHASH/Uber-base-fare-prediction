import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Uber Fare Prediction AI", layout="centered")

st.title("🚕 Uber Fare Prediction")
st.write("Predict the estimated fare of an Uber ride using Machine Learning.")

# -----------------------------------
# Load Trained Model
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("uber_fare_model.pkl")

model = load_model()

# -----------------------------------
# User Inputs
# -----------------------------------

st.subheader("Enter Ride Details")

pickup_longitude = st.number_input("Pickup Longitude", value=-73.985428)
pickup_latitude = st.number_input("Pickup Latitude", value=40.748817)

dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.985428)
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.748817)

passenger_count = st.slider("Passenger Count",1,6,1)

hour = st.slider("Hour of Day",0,23,12)
day = st.slider("Day of Month",1,31,15)
month = st.slider("Month",1,12,6)
weekday = st.slider("Weekday (0=Mon)",0,6,3)

# -----------------------------------
# Calculate Distance (Haversine)
# -----------------------------------

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R*c

distance = haversine(
    pickup_latitude,
    pickup_longitude,
    dropoff_latitude,
    dropoff_longitude
)

st.write(f"Estimated Distance: **{distance:.2f} km**")

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict Fare"):

    input_data = pd.DataFrame([[
        pickup_longitude,
        pickup_latitude,
        dropoff_longitude,
        dropoff_latitude,
        passenger_count,
        hour,
        day,
        month,
        weekday,
        distance
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
        "distance_km"
    ])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Uber Fare: ${prediction[0]:.2f}")
