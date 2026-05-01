import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
