import streamlit as st
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier

def run_prediction():
    st.title("Breast Cancer Predictor")

    # Load or train model
    @st.cache_resource
    def load_model():
        data = fetch_ucirepo(id=17)
        X = data.data.features
        y = data.data.targets['diagnosis'].map({'M': 1, 'B': 0})
        model = RandomForestClassifier()
        model.fit(X, y)
        return model

    model = load_model()
    st.write("Input values for prediction:")

    input_data = {}
    for col in fetch_ucirepo(id=17).data.features.columns[:5]:  # Show only 5 for demo
        input_data[col] = st.number_input(col, value=1.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        label = "Malignant" if pred == 1 else "Benign"
        st.success(f"The predicted diagnosis is: **{label}**")
