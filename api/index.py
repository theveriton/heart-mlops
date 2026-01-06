from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import logging
import sklearn
from prometheus_client import Counter, generate_latest

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the fitted pipeline (preprocessing + model)
model = joblib.load("model.pkl")
model_info = {
    "model_type": type(model).__name__,
    "sklearn_version": sklearn.__version__
}

# Prometheus metrics
predict_requests = Counter('predict_requests_total', 'Total number of prediction requests')


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/info")
def info():
    details = model_info.copy()
    try:
        inner = None
        has_multi = False
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner = type(model.named_steps["model"]).__name__
            has_multi = hasattr(model.named_steps["model"], "multi_class")
        details.update({"inner_model": inner, "has_multi_class_attr": has_multi})
    except Exception as e:
        details.update({"inner_error": str(e)})
    return details


@app.get("/metrics")
def metrics():
    return generate_latest()


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    predict_requests.inc()  # Increment Prometheus counter
    
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