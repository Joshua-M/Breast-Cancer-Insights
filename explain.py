import streamlit as st
import shap
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier

def run_explanation():
    st.title("Model Explanation")

    data = fetch_ucirepo(id=17)
    X = data.data.features
    y = data.data.targets['diagnosis'].map({'M': 1, 'B': 0})

    model = RandomForestClassifier()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.write("### Feature Importance (SHAP Summary)")
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

    st.write("### SHAP Value Explanation (first instance)")
    shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.iloc[0,:], matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')
