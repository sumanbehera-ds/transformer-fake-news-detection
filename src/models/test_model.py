from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

MODEL_PATH = "models/final_roberta_fake_news"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

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

label_map = {
    0: "FAKE",
    1: "REAL"
}

print("Prediction:", label_map[prediction])
print("Confidence:", probabilities[0][prediction].item())