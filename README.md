🚕 Uber Fare Prediction using Machine Learning

This project predicts the fare amount of Uber rides using machine learning regression models. The model learns from historical trip data such as pickup and drop-off locations, trip distance, time features, and passenger count to estimate the expected fare.

The goal of this project is to demonstrate data preprocessing, feature engineering, visualization, and regression modeling for real-world transportation data.

📊 Dataset

The dataset contains historical Uber ride information including:

Pickup and drop-off longitude & latitude

Pickup date and time

Passenger count

Fare amount

Using these features, the model predicts the price of a ride.

⚙️ Project Workflow

The project follows a typical machine learning pipeline:

1️⃣ Data Loading

Load Uber dataset using Pandas.

2️⃣ Data Exploration

Inspect dataset structure

Check data types

Analyze statistical summary

3️⃣ Data Cleaning

Remove unnecessary columns

Handle missing values

Remove invalid coordinates

Filter unrealistic fare values

Remove incorrect passenger counts

4️⃣ Feature Engineering

New features are created from the pickup datetime:

Hour

Day

Month

Weekday

Trip distance is calculated using the Haversine Formula from pickup and drop-off coordinates.

5️⃣ Data Visualization

Correlation heatmap

Scatter plots between distance and fare

Outlier detection

6️⃣ Model Training

Three regression models were trained:

Linear Regression

Random Forest Regressor

XGBoost Regressor

7️⃣ Model Evaluation

Models are evaluated using:

RMSE (Root Mean Squared Error)

R² Score

🧠 Machine Learning Models Used
Model	Purpose
Linear Regression	Baseline regression model
Random Forest	Handles nonlinear relationships
XGBoost	Gradient boosting for better performance
🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Google Colab

📈 Key Feature: Distance Calculation

The distance between pickup and drop-off locations is computed using the Haversine formula, which calculates the shortest distance between two points on the Earth.

This feature plays a major role in predicting fare prices.

📂 Project Structure
Uber-Fare-Prediction
  uber.csv
  Uber_Fare_Prediction.ipynb
  uber_fare_model.pkl
  README.md

🎯 Future Improvements

Add deep learning models

Deploy the model using Flask or FastAPI

Build a user interface for fare prediction

Integrate real-time maps API
