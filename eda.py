# eda.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model_utils import load_clean_data

def run_eda():
    st.title("📊 Exploratory Data Analysis")

    # Load cleaned dataset
    df = load_clean_data()

    st.subheader("🗃 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📏 Data Summary")
    st.dataframe(df.describe())

    st.subheader("🧮 Class Distribution")
    class_counts = df["class"].value_counts().sort_index()
    class_labels = {2: "Benign", 4: "Malignant"}
    labeled_counts = class_counts.rename(index=class_labels)
    st.bar_chart(labeled_counts)
    st.info(f"🔍 There are **{labeled_counts['Malignant']} malignant** and **{labeled_counts['Benign']} benign** cases.")

    st.subheader("📌 Missing Values")
    missing = df.isnull().sum()
    if missing.any():
        st.warning("There are missing values in the dataset:")
        st.write(missing[missing > 0])
    else:
        st.success("No missing values detected after cleaning.")

    st.subheader("📊 Feature Distributions by Class")
    feature_cols = [col for col in df.columns if col not in ["class"]]
    default_features = feature_cols[:2] if len(feature_cols) >= 2 else feature_cols
    selected_features = st.multiselect("Select features to compare:", feature_cols, default=default_features)

    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.boxplot(x='class', y=feature, data=df.replace({'class': {2: 'Benign', 4: 'Malignant'}}), ax=ax)
        ax.set_title(f'{feature.replace("_", " ").title()} by Class')
        st.pyplot(fig)

    st.subheader("📈 Correlation Heatmap")
    # Drop ID column if present
    if "sample_code_number" in df.columns:
        corr = df.drop(columns=["sample_code_number"]).corr()
    else:
        corr = df.corr()

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🔎 Key Insights")
    st.markdown("""
    - **Clump Thickness**, **Cell Size Uniformity**, and **Bare Nuclei** show strong separation between benign and malignant tumours.
    - Benign cases (`2`) are more frequent than malignant (`4
