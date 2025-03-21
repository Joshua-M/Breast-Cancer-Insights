import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("2025-03-21T13-25_export.csv")
    return df

def run_eda():
    st.title("📊 Exploratory Data Analysis")

    df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    st.bar_chart(df["Diagnosis"].value_counts())

    st.subheader("Correlation Heatmap (First 10 features)")
    numeric_df = df.select_dtypes(include='number').drop(columns=['Unnamed: 0'])
    corr = numeric_df.iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplots (Area1 by Diagnosis)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="Diagnosis", y="area1", ax=ax2)
    st.pyplot(fig2)
