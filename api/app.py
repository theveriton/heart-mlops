from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("model.pkl","rb"))

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: list):
    arr = np.array(data).reshape(1,-1)
    prob = model.predict_proba(arr)[0,1]
    return {"risk_score": float(prob)}
