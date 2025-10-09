import streamlit as st
import pandas as pd
import pickle

st.title("Car Price Prediction App")

# --- Load model safely ---
try:
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Upload CSV and predict safely ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())

        preds = model.predict(df)
        df['Predicted Price'] = preds
        st.write("Predictions:")
        st.dataframe(df[['Predicted Price']])
    except Exception as e:
        st.error(f"Error during prediction: {e}")

