# explain.py
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from model_utils import train_model, load_clean_data

def run_explanation():
    st.title("🧠 Model Explainability")

    model, X_train, X_test, y_train, y_test = train_model()

    st.subheader("📊 Global Feature Importance (SHAP)")
    st.markdown("Understanding how each feature impacts predictions globally.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("🔍 SHAP Summary Plot (Distribution)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🔬 Individual Prediction Explanation (Force Plot)")

    sample_index = st.slider("Select a test sample index", 0, len(X_test)-1, 0)
    shap_values_test = explainer.shap_values(X_test)

    st.markdown("Force plot for sample below:")

    # ✅ Fix: Ensure expected_value is a scalar
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):  # If multi-class, pick the relevant class
        expected_value = expected_value[1]  # Selecting class 1 (malignant)

    # ✅ Fix: Handle SHAP values for different model types
    if isinstance(shap_values_test, list):
        shap_force_values = shap_values_test[1]  # Use index 1 for class 1 (malignant)
    else:
        shap_force_values = shap_values_test

    # ✅ Convert Pandas Series to NumPy array for force_plot()
    feature_values = X_test.iloc[sample_index].values.reshape(1, -1)

    fig3 = shap.force_plot(
        expected_value,
        shap_force_values[sample_index],
        feature_values,  # ✅ Ensure correct format
        matplotlib=True
    )
    st.pyplot(fig3)

    st.markdown("---")
    st.info("Features to the right increase the chance of a **malignant** diagnosis, while those to the left push toward **benign**.")
