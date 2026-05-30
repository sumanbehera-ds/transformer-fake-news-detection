import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fake News Detection API",
    description="Fake news classifier using Hugging Face Inference API",
    version="1.0.0"
)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sumanbehera-ds/roberta-fake-news-detector"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class NewsInput(BaseModel):
    text: str = Field(..., min_length=5)

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_source": "Hugging Face Inference API",
        "model_id": MODEL_ID
    }

@app.post("/predict")
def predict_news(data: NewsInput):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": data.text}
    )

    result = response.json()

    if isinstance(result, dict) and "error" in result:
        return {
            "error": result["error"]
        }

    scores = result[0]

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