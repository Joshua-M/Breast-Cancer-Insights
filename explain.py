import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("2025-03-21T13-25_export.csv")
    return df

def run_explanation():
    st.title("🧠 Model Explanation")

    df = load_data()
    X = df.drop(columns=["Unnamed: 0", "Diagnosis"])
    y = df["Diagnosis"].map({"M": 1, "B": 0})

    model = RandomForestClassifier()
    model.fit(X, y)

    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig1 = shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

    st.subheader("SHAP Force Plot (first observation)")
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0], matplotlib=True)
    st.pyplot(bbox_inches='tight')
