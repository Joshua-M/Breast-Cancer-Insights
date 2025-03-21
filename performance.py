# performance.py
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from model_utils import (
    train_model,
    evaluate_model,
    plot_actual_vs_predicted_scatter,
)

def run_performance():
    st.title("📈 Model Performance")

    # Model selection
    model_type = st.selectbox("Select Model to Evaluate", ["Random Forest", "Logistic Regression", "Decision Tree"])
    model, X_train, X_test, y_train, y_test = train_model(model_type)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)

    # Metrics
    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['accuracy']:.2%}")
    col2.metric("Precision", f"{results['precision']:.2%}")
    col3.metric("Recall", f"{results['recall']:.2%}")
    col4.metric("F1 Score", f"{results['f1']:.2%}")

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, results["y_pred"])
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(ax=ax_cm, cmap="Blues")
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("📑 Classification Report")
    report = classification_report(y_test, results["y_pred"], target_names=["Benign", "Malignant"])
    st.code(report)

    # Scatter Plot: Actual vs Predicted
    st.subheader("📈 Actual vs Predicted (Scatter Plot)")
    fig_scatter = plot_actual_vs_predicted_scatter(y_test, results["y_pred"])
    st.pyplot(fig_scatter)
