# explain.py
import streamlit as st
import shap
import matplotlib.pyplot as plt
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
    shap.initjs()
    fig3 = shap.force_plot(
        explainer.expected_value,
        shap_values_test[sample_index],
        X_test.iloc[sample_index],
        matplotlib=True
    )
    st.pyplot(fig3)

    st.markdown("---")
    st.info("Features to the right increase the chance of a **malignant** diagnosis, while those to the left push toward **benign**.")
