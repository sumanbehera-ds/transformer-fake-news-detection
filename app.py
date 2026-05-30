import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fake News Detection API")

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sumanbehera-ds/roberta-fake-news-detector"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class NewsInput(BaseModel):
    text: str = Field(..., min_length=5)

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_id": MODEL_ID}

@app.post("/predict")
def predict_news(data: NewsInput):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": data.text},
        timeout=60
    )

    result = response.json()

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=result)

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    if not isinstance(result, list):
        raise HTTPException(status_code=500, detail={"unexpected_output": result})

    scores = result[0]

    if isinstance(scores, dict):
        scores = [scores]

    best = max(scores, key=lambda x: x["score"])

    label_map = {
        "LABEL_0": "FAKE",
        "LABEL_1": "REAL"
    }

    return {
        "prediction": label_map.get(best["label"], best["label"]),
        "confidence": round(best["score"], 4),
        "raw_output": scores
    }