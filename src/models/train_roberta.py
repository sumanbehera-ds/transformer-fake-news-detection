import pandas as pd
import numpy as np
import mlflow
import torch

from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

MODEL_NAME = "roberta-base"

COLS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_fire_counts", "context"
]

def convert_label(label):
    return 1 if label in ["true", "mostly-true"] else 0

train_df = pd.read_csv(
    "data/raw/train.tsv",
    sep="\t",
    header=None,
    names=COLS
)

valid_df = pd.read_csv(
    "data/raw/valid.tsv",
    sep="\t",
    header=None,
    names=COLS
)

train_df = train_df.rename(columns={"label": "labels"})
valid_df = valid_df.rename(columns={"label": "labels"})

train_ds = Dataset.from_pandas(train_df[["statement", "labels"]])
valid_ds = Dataset.from_pandas(valid_df[["statement", "labels"]])

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["statement"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = Dataset.from_pandas(
    train_df[["statement", "label"]]
)

valid_ds = Dataset.from_pandas(
    valid_df[["statement", "label"]]
)

train_ds = train_ds.map(tokenize, batched=True)
valid_ds = valid_ds.map(tokenize, batched=True)

train_ds.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

valid_ds.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    probs = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()[:, 1]

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs)
    }

training_args = TrainingArguments(
    output_dir="models/roberta_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics
)

mlflow.set_experiment("fake_news_roberta")

with mlflow.start_run():

    trainer.train()

    metrics = trainer.evaluate()

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    model.save_pretrained("models/final_roberta_fake_news")
    tokenizer.save_pretrained("models/final_roberta_fake_news")

    mlflow.log_param("model", "roberta-base")
    mlflow.log_param("epochs", 2)

print("Training completed.")