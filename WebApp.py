import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained linear_regression_model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

# Streamlit App Title
st.title("🚗 Car Price Prediction App")

st.markdown("Welcome to the **Car Price Prediction App**! Enter your car details and get an estimated price.")

# Sidebar for user input
st.sidebar.header("Enter Car Details")
year = st.sidebar.slider("Manufacturing Year", 2000, 2024, 2015)
km_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 50000)
mileage = st.sidebar.number_input("Mileage (kmpl)", 5.0, 30.0, 18.0)
engine = st.sidebar.number_input("Engine Capacity (CC)", 500, 5000, 1200)
max_power = st.sidebar.number_input("Max Power (bhp)", 30, 500, 100)
seats = st.sidebar.slider("Number of Seats", 2, 8, 5)

# Convert input into DataFrame format
features = np.array([[year, km_driven, mileage, engine, max_power, seats]])

# Predict price
if st.sidebar.button("Predict Price"):
    prediction = model.predict(features)
    st.write(f"### 💰 Estimated Selling Price: ₹{round(prediction[0], 2)}")

st.write("🔍 Adjust values in the sidebar and click **Predict Price** to see results!")