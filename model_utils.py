# model_utils.py

import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_clean_data():
    data = fetch_ucirepo(id=15)
    df = pd.concat([data.data.features, data.data.targets], axis=1)

    df.columns = df.columns.str.strip()
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    for col in df.columns:
        if col != 'class':
            df[col] = pd.to_numeric(df[col], errors='ignore')

    df['class'] = df['class'].astype(int)

    return df

@st.cache_resource
def train_model(model_type="Random Forest"):
    df = load_clean_data()

    drop_cols = ["class"]
    if "sample_code_number" in df.columns:
        drop_cols.append("sample_code_number")

    X = df.drop(columns=drop_cols)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=4)
    rec = recall_score(y_test, y_pred, pos_label=4)
    f1 = f1_score(y_test, y_pred, pos_label=4)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred
    }

def plot_prediction_line_chart(y_test, y_pred):
    df = pd.DataFrame({
        "Index": range(len(y_test)),
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Index"], df["Actual"], label="Actual", marker='o', linestyle='-')
    ax.plot(df["Index"], df["Predicted"], label="Predicted", marker='x', linestyle='--')
    ax.set_yticks([2, 4])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_title("Actual vs Predicted Tumour Class (Line Chart)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class")
    ax.legend()
    return fig

