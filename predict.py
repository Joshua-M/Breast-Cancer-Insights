# predict.py
import streamlit as st
import pandas as pd
from model_utils import train_model, evaluate_model, plot_prediction_comparison, load_clean_data

def run_predictor():
    st.title("🔮 Breast Cancer Prediction")

    # Model selection
    st.subheader("⚙️ Choose Model")
    model_type = st.selectbox("Select a model type:", ["Random Forest", "Logistic Regression", "Decision Tree"])

    model, X_train, X_test, y_train, y_test = train_model(model_type)
    feature_columns = X_train.columns

    st.subheader("📋 Input Features")
    st.markdown("Use manual entry for one prediction or upload a file for batch predictions.")

    input_method = st.radio("Choose input method:", ["📝 Manual Entry", "📂 Upload CSV"])

    if input_method == "📝 Manual Entry":
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
    else:
        uploaded_file = st.file_uploader("Upload a CSV with the correct features", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        else:
            st.warning("Please upload a CSV file to proceed.")
            return

    st.subheader("🔎 Prediction Result")

    if st.button("Predict"):
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        for i, pred in enumerate(predictions):
            label = "Malignant" if pred == 4 else "Benign"
            conf = probabilities[i][1 if pred == 4 else 0]
            st.success(f"Prediction {i+1}: **{label}** (Confidence: {conf:.2%})")

        # Evaluate model and show metrics if batch file used
        if input_method == "📂 Upload CSV" and input_df.shape[0] > 1:
            df = load_clean_data()
            drop_cols = ["class"]
            if "sample_code_number" in df.columns:
                drop_cols.append("sample_code_number")

            y_true = df["class"].iloc[:len(input_df)]  # use real y values if available
            eval_result = evaluate_model(model, input_df, y_true)
            st.subheader("📊 Model Performance on Uploaded Data")
            st.metric("Accuracy", f"{eval_result['accuracy']:.2%}")
            st.metric("Precision", f"{eval_result['precision']:.2%}")
            st.metric("Recall", f"{eval_result['recall']:.2%}")
            st.metric("F1 Score", f"{eval_result['f1']:.2%}")

            fig = plot_prediction_comparison(y_true, eval_result["y_pred"])
            st.pyplot(fig)
