from fastapi import FastAPI
from detect_anomaly import detect
import pandas as pd

app = FastAPI()

@app.post("/detect")

def detect_intrusion(data: dict):

    df = pd.DataFrame([data])

    result = detect(df)

    return {"result": result}