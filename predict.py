import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("2025-03-21T13-25_export.csv")
    return df

def run_prediction():
    st.title("🔮 Breast Cancer Predictor")

    df = load_data()
    X = df.drop(columns=["Unnamed: 0", "Diagnosis"])
    y = df["Diagnosis"].map({"M": 1, "B": 0})

    model = RandomForestClassifier()
    model.fit(X, y)

    st.write("### Input Feature Values")
    user_input = {}
    for col in X.columns[:5]:  # Show a few for demo
        user_input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        label = "Malignant" if prediction == 1 else "Benign"
        st.success(f"Predicted Diagnosis: **{label}**")
