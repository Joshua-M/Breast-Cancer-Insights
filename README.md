# 🧠 Breast Cancer Insights

A Streamlit-powered machine learning dashboard that predicts whether a breast tumour is benign or malignant based on the **Breast Cancer Wisconsin (Original) dataset**.

🔗 **Live App:**  
👉 [https://breast-cancer-insights-nxe4yvvwv7lqzvv4q3hkte.streamlit.app/](https://breast-cancer-insights-nxe4yvvwv7lqzvv4q3hkte.streamlit.app/)

---

## 📌 Overview

This interactive tool allows users to:
- Explore the dataset through visualisations
- Manually enter tumour cell features and predict diagnosis
- Evaluate model performance with accuracy, precision, recall and confusion matrix

---

## 📊 Sections
- **📊 Explore the Data (EDA):**  
  Visualise distributions, class balance, and correlations.

- **🔮 Predict Tumour Type from Cell Data:**  
  Input real or hypothetical patient measurements and get instant predictions.

- **📈 Evaluate Model Performance:**  
  View metrics like accuracy, F1 score, and see actual vs predicted classifications.

---

## 📦 Tech Stack
- **Streamlit** – for building the dashboard UI
- **scikit-learn** – for training models
- **SHAP (optional)** – for model explainability
- **Matplotlib & Seaborn** – for plotting
- **Pandas & NumPy** – for data manipulation
- **ucimlrepo** – to fetch the UCI dataset programmatically

---

## 📁 Dataset
**Breast Cancer Wisconsin (Original)**  
🔗 [UCI ML Repository – Dataset #15](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

- Features describe characteristics of cell nuclei present in digitised images of breast fine needle aspirates (FNAs).
- Binary classification:
  - `2` = Benign
  - `4` = Malignant

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Joshua-M/breast-cancer-insights.git
cd breast-cancer-insights

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
