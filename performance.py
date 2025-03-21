# performance.py
import streamlit as st
from model_utils import train_model, evaluate_model, plot_prediction_comparison
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def run_performance():
    st.title("📈 Model Performance")

    # Choose model
    model_type = st.selectbox("Select Model to Evaluate", ["Random Forest", "Logistic Regression", "Decision Tree"])
    model, X_train, X_test, y_train, y_test = train_model(model_type)
    
    results = evaluate_model(model, X_test, y_test)

    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['accuracy']:.2%}")
    col2.metric("Precision", f"{results['precision']:.2%}")
    col3.metric("Recall", f"{results['recall']:.2%}")
    col4.metric("F1 Score", f"{results['f1']:.2%}")

    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, results["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm, cmap="Blues")
    st.pyplot(fig_cm)

    st.subheader("📑 Classification Report")
    report = classification_report(y_test, results["y_pred"], target_names=["Benign", "Malignant"])
    st.code(report)

    st.subheader("🧾 Actual vs Predicted Chart")
    fig_bar = plot_prediction_comparison(y_test, results["y_pred"])
    st.pyplot(fig_bar)
