import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fake News Detection API")

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sumanbehera-ds/roberta-fake-news-detector"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class NewsInput(BaseModel):
    text: str = Field(..., min_length=5)

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_id": MODEL_ID}

@app.get("/debug")
def debug():
    return {
        "hf_token_exists": HF_TOKEN is not None,
        "model_id": MODEL_ID,
        "api_url": API_URL
    }

@app.post("/predict")
def predict_news(data: NewsInput):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": data.text},
            timeout=60
        )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        result = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=result)

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

    except Exception as e:
        print("PREDICT ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))