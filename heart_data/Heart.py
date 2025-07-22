import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Title
st.title("💡 Machine Learning Project - Heart Disease Classification")

# Load model
try:
    model = joblib.load('D:\p\model_heart.pt')
except FileNotFoundError:
    st.error("❌ Model file 'D:\p\model_heart.pt' not found. Please upload it.")

# File uploader
st.header("📁 Upload CSV File for Prediction")
file = st.file_uploader("Upload your CSV file (must have required features)", type='csv')

if file is not None:
    try:
        # Read file
        data = pd.read_csv(file)
        st.subheader("🔍 Uploaded Data Preview")
        st.write(data.head())

        # Prediction
        st.subheader("✅ Prediction Results")
        prediction = model.predict(data)
        data['Predicted'] = prediction
        st.write(data)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

else:
    st.info("👆 Please upload a CSV file with the correct format to see predictions.")
