# model_utils.py
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_clean_data():
    data = fetch_ucirepo(id=15)
    df = pd.concat([data.data.features, data.data.targets], axis=1)
    df.columns = df.columns.str.strip()

    # Clean column names
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    # Replace '?' with NaN and convert to float where possible
    df.replace('?', pd.NA, inplace=True)
    df = df.dropna()
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)

    df['class'] = df['class'].astype(int)
    return df

@st.cache_resource
def train_model():
    df = load_clean_data()
    X = df.drop(columns=["sample_code_number", "class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
