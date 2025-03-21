# predict.py
import streamlit as st
import pandas as pd
from model_utils import (
    train_model,
    evaluate_model,
    plot_actual_vs_predicted_scatter,  # ✅ Correct function name
    load_clean_data
)

def run_predictor():
    st.title("🔮 Breast Cancer Predictor")

    st.subheader("⚙️ Model Selection")
    model_type = st.selectbox("Choose a model to use:", ["Random Forest", "Logistic Regression", "Decision Tree"])

    model, X_train, X_test, y_train, y_test = train_model(model_type)
    feature_columns = X_train.columns

    st.subheader("📝 Enter Feature Values")
    st.markdown("Input tumour-related measurements below to get a prediction.")

    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_columns):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(X_train[feature].mean())
            )

    input_df = pd.DataFrame([input_data])

    st.subheader("🔎 Prediction")

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        label = "Malignant" if prediction == 4 else "Benign"
        confidence = probabilities[1 if prediction == 4 else 0]

        st.success(f"**Predicted Diagnosis: {label}**")
        st.info(f"Prediction Confidence: **{confidence:.2%}**")

    st.markdown("---")
    st.subheader("📈 Prediction Trend (Test Set)")
    st.markdown("Visual comparison between actual and predicted tumour classes in the test set.")

    results = evaluate_model(model, X_test, y_test)
    fig = plot_actual_vs_predicted_scatter(y_test, results["y_pred"])  # ✅ Fixed function call
    st.pyplot(fig)
