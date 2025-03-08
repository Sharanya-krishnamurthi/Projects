import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Salary Prediction App")

years_of_experience = st.number_input(
    "Enter your Years of Experience:", min_value=0.0, step=0.1
)

if st.button("Predict Salary"):
    input_data = np.array([[years_of_experience]])

    predicted_salary = model.predict(input_data)[0]

    st.header(f"Estimated Salary: ${predicted_salary:,.2f}")
