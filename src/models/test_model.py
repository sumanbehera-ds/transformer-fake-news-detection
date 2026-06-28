import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "models/final_roberta_fake_news"
REMOTE_MODEL_ID = "sumanbehera-ds/roberta-fake-news-detector"


def resolve_model_source():
    local_path = Path(MODEL_PATH)
    has_local_weights = (
        (local_path / "model.safetensors").exists()
        or (local_path / "pytorch_model.bin").exists()
    )
    return MODEL_PATH if has_local_weights else os.getenv("HF_MODEL_ID", REMOTE_MODEL_ID)


model_source = resolve_model_source()
tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source)

model.eval()

text = "The government secretly controls the weather."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

label_map = getattr(model.config, "id2label", None) or {
    0: "FAKE",
    1: "REAL"
}
prediction_label = label_map.get(prediction, label_map.get(str(prediction), str(prediction)))

print("Model source:", model_source)
print("Prediction:", prediction_label)
print("Confidence:", probabilities[0][prediction].item())
