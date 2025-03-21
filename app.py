import streamlit as st
from eda import run_eda
from predict import run_prediction
from explain import run_explanation

st.set_page_config(page_title="Breast Cancer Insights Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔮 Predictor", "🧠 Model Explanation"])

if page == "🏠 Home":
    st.title("Breast Cancer Insights")
    st.write("""
    Welcome! This dashboard helps you:
    - Explore breast cancer data visually (EDA)
    - Predict diagnosis (Malignant or Benign)
    - Understand predictions using SHAP explanations
    """)
elif page == "📊 EDA":
    run_eda()
elif page == "🔮 Predictor":
    run_prediction()
elif page == "🧠 Model Explanation":
    run_explanation()
