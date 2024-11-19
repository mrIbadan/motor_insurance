import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Generate Synthetic Data
def generate_synthetic_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "Age": np.random.randint(18, 75, 1000),
        "Annual Mileage": np.random.randint(5000, 30000, 1000),
        "Vehicle Age": np.random.randint(0, 20, 1000),
        "Vehicle Value": np.random.randint(5000, 50000, 1000),
        "Location Risk Score": np.random.uniform(0.5, 2.0, 1000),
        "Policy Type": np.random.choice(["Comprehensive", "Third-Party"], 1000),
    })
    base_rate = 200
    data["Premium"] = (
        base_rate +
        0.3 * (75 - data["Age"]) +
        0.0005 * data["Annual Mileage"] +
        0.5 * data["Vehicle Age"] +
        0.02 * data["Vehicle Value"] +
        50 * data["Location Risk Score"] +
        (data["Policy Type"] == "Comprehensive") * 100
    )
    return data

# Step 2: Train a Simple GLM Model
data = generate_synthetic_data()
features = ["Age", "Annual Mileage", "Vehicle Age", "Vehicle Value", "Location Risk Score", "Policy Type"]
data_encoded = pd.get_dummies(data, columns=["Policy Type"], drop_first=True)
X = data_encoded.drop(columns=["Premium"])
y = data_encoded["Premium"]

model = LinearRegression()
model.fit(X, y)

# Step 3: Streamlit App
st.title("Car Insurance Rating Engine")

# User Form for Input
st.header("Enter Customer Information")
age = st.slider("Driver's Age", 18, 75, 30)
annual_mileage = st.number_input("Annual Mileage (km)", min_value=5000, max_value=30000, step=1000, value=15000)
vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
vehicle_value = st.number_input("Vehicle Value ($)", min_value=5000, max_value=50000, step=1000, value=20000)
location_risk = st.slider("Location Risk Score (1 = Low Risk, 2 = High Risk)", 0.5, 2.0, 1.0)
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third-Party"])

# Predict Premium
if st.button("Calculate Premium"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Annual Mileage": [annual_mileage],
        "Vehicle Age": [vehicle_age],
        "Vehicle Value": [vehicle_value],
        "Location Risk Score": [location_risk],
        "Policy Type_Third-Party": [1 if policy_type == "Third-Party" else 0],
    })
    predicted_premium = model.predict(input_data)[0]
    st.subheader("Quoted Premium")
    st.write(f"The estimated premium is **£{predicted_premium:.2f}**.")
