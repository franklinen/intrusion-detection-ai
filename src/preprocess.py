import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_preprocess(path):

    df = pd.read_csv(path)

    categorical = ['proto', 'service', 'state']

    encoders = {}

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(['label'], axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y