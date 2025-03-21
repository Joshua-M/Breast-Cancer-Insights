# 🧠 Breast Cancer Insights Dashboard

An interactive Streamlit dashboard for exploring, predicting, and explaining breast cancer diagnoses using the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## 📦 Features

- **📊 Exploratory Data Analysis (EDA)**  
  View class distributions, summary statistics, correlation heatmaps, and pairplots.

- **🔮 Prediction Tool**  
  Enter values or upload a CSV to predict if a tumour is *Malignant* or *Benign* using a trained Random Forest classifier.

- **🧠 Model Explainability**  
  Understand model behaviour with feature importance charts and SHAP value visualisations.

---

## 📁 Dataset

This app uses the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) dataset from the UCI Machine Learning Repository, accessed via the `ucimlrepo` package.

- 30 real-valued features (mean, standard error, worst of each of 10 cell nuclei measurements)
- 569 instances
- Binary classification:  
  - `M` = Malignant  
  - `B` = Benign

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Joshua-M/breast-cancer-insights.git
cd breast-cancer-insights
