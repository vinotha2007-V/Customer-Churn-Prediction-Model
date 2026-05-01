import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('../models/churn_model.pkl', 'rb'))

st.title("Customer Churn Prediction")

features = []
for i in range(19):
    val = st.number_input(f"Feature {i+1}")
    features.append(val)

if st.button("Predict"):
    prediction = model.predict([features])
    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will stay ✅")
