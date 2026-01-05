from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the fitted pipeline (preprocessing + model)
model = joblib.load("model.pkl")

# Simple metrics
predict_requests = 0


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/metrics")
def metrics():
    return {"predict_requests": predict_requests}


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    global predict_requests
    predict_requests += 1
    
    features = req.features
    df = pd.DataFrame([features])
    try:
        prob = model.predict_proba(df)[0, 1]
        pred = int(prob >= 0.5)
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))

    # Log request summary
    logging.info("Predict request from %s: pred=%s, score=%.4f", request.client.host if request.client else "local", pred, float(prob))

    return {"prediction": pred, "risk_score": float(prob)}


# Vercel expects the app to be named 'app'
# This is the entry point for Vercel
handler = app