import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

def run_eda():
    st.title("Exploratory Data Analysis")

    data = fetch_ucirepo(id=17)
    df = pd.concat([data.data.features, data.data.targets], axis=1)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Class Distribution")
    st.bar_chart(df['diagnosis'].value_counts())

    st.write("### Correlation Heatmap")
    corr = df.drop(columns=["diagnosis"]).corr()
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

    st.write("### Pairplot (first few features)")
    selected = df[['radius_mean', 'texture_mean', 'area_mean', 'diagnosis']]
    fig2 = sns.pairplot(selected, hue="diagnosis")
    st.pyplot(fig2)
