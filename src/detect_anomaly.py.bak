import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model("models/lstm_autoencoder.h5")
scaler = joblib.load("models/scaler.pkl")

THRESHOLD = 0.08

def detect(data):

    data_scaled = scaler.transform(data)

    data_scaled = np.expand_dims(data_scaled, axis=0)

    reconstruction = model.predict(data_scaled)

    mse = np.mean(np.power(data_scaled - reconstruction, 2))

    if mse > THRESHOLD:
        return "ANOMALY"
    else:
        return "NORMAL"