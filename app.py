import streamlit as st
from eda import run_eda
from predict import run_prediction
from explain import run_explanation

st.set_page_config(page_title="Breast Cancer Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔮 Predictor", "🧠 Model Explanation"])

if page == "🏠 Home":
    st.title("Breast Cancer Wisconsin Diagnostic Dataset")
    st.write("""
    This dashboard offers:
    - Exploratory Data Analysis (EDA)
    - Prediction of breast cancer type (Malignant or Benign)
    - Model Explanation using Feature Importance and SHAP
    """)
elif page == "📊 EDA":
    run_eda()
elif page == "🔮 Predictor":
    run_prediction()
elif page == "🧠 Model Explanation":
    run_explanation()
