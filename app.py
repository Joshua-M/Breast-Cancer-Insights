# app.py
import streamlit as st
from eda import run_eda
from predict import run_predictor
from performance import run_performance
from explain import run_explanation

st.set_page_config(page_title="Breast Cancer Wisconsin (Original)", layout="wide")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔮 Predictor", "📈 Performance"])

if page == "🏠 Home":
    st.title("🧬 Breast Cancer Wisconsin (Original)")
    st.markdown("""
    Welcome to the **Breast Cancer Insights Dashboard**.

    This app is built using the UCI Breast Cancer Wisconsin (Original) dataset.
    
    **Sections:**
    - 📊 Explore the data (EDA)
    - 🔮 Predict tumour type from cell data
    - 📈 Evaluate model performance

    **Target:**
    - `2`: Benign
    - `4`: Malignant

    **Source:** [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
    """)
elif page == "📊 EDA":
    run_eda()
elif page == "🔮 Predictor":
    run_predictor()
elif page == "📈 Performance":
    run_performance()    

