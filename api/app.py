from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import logging
import sklearn

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the fitted pipeline (preprocessing + model)
model = joblib.load("model.pkl")
model_info = {
    "model_type": type(model).__name__,
    "sklearn_version": sklearn.__version__
}

# Simple metrics
predict_requests = 0


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/info")
def info():
    return model_info


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
