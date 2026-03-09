import shap
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

model = load_model("models/lstm_autoencoder.h5")

background = np.random.normal(size=(100,10,40))

explainer = shap.DeepExplainer(model, background)

def explain(sample):

    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample)