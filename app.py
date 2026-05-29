from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = FastAPI(
    title="Fake News Detection API",
    description="RoBERTa-based fake news classifier",
    version="1.0.0"
)

MODEL_PATH = "sumanbehera-ds/roberta-fake-news-detector"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {
    0: "FAKE",
    1: "REAL"
}

class NewsInput(BaseModel):
    text: str = Field(..., min_length=5, example="The government secretly controls the weather.")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "RoBERTa Fake News Classifier"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_news(data: NewsInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return {
        "prediction": label_map[prediction],
        "confidence": round(confidence, 4)
    }