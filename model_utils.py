# model_utils.py

import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_clean_data():
    # Fetch from UCI ML Repo
    data = fetch_ucirepo(id=15)
    df = pd.concat([data.data.features, data.data.targets], axis=1)

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    # Replace '?' with NaN and drop rows with missing values
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert feature columns to numeric where necessary
    for col in df.columns:
        if col != 'class':
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # Make sure class column is int
    df['class'] = df['class'].astype(int)

    return df

@st.cache_resource
def train_model():
    df = load_clean_data()

    # Safely drop optional ID column if it exists
    drop_cols = ["class"]
    if "sample_code_number" in df.columns:
        drop_cols.append("sample_code_number")

    X = df.drop(columns=drop_cols)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
