# predict.py
import streamlit as st
import pandas as pd
from model_utils import train_model, load_clean_data

def run_predictor():
    st.title("🔮 Breast Cancer Prediction")

    model, X_train, _, _, _ = train_model()
    feature_columns = X_train.columns

    st.subheader("📋 Input Features")
    st.markdown("Adjust values below or upload a file to predict tumour class.")

    option = st.radio("Choose input method:", ["📝 Manual Entry", "📂 Upload CSV"])

    if option == "📝 Manual Entry":
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_columns):
            with cols[i % 3]:
                val = st.number_input(feature.replace("_", " ").title(), min_value=0.0, value=float(X_train[feature].mean()))
                input_data[feature] = val
        input_df = pd.DataFrame([input_data])

    else:
        uploaded_file = st.file_uploader("Upload a CSV file with correct feature names.", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        else:
            st.warning("Please upload a file to proceed.")
            return

    st.subheader("🔎 Prediction Result")

    if st.button("Predict"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        for i in range(len(input_df)):
            result = "Malignant" if prediction[i] == 4 else "Benign"
            st.info(f"**Prediction {i+1}: {result}**")
            st.progress(float(proba[i][prediction[i] == model.classes_]))
