import os

import mlflow
import numpy as np
import pandas as pd
import torch

from torch import nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
CLASS_WEIGHTS = [1.0, 1.8]

COLS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_fire_counts", "context"
]


def convert_label(label):
    return 1 if label in ["true", "mostly-true"] else 0


def load_data():
    train_df = pd.read_csv("data/raw/train.tsv", sep="\t", header=None, names=COLS)
    valid_df = pd.read_csv("data/raw/valid.tsv", sep="\t", header=None, names=COLS)

    train_df["labels"] = train_df["label"].apply(convert_label)
    valid_df["labels"] = valid_df["label"].apply(convert_label)

    train_df = train_df[["statement", "labels"]]
    valid_df = valid_df[["statement", "labels"]]

    return train_df, valid_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs)
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        class_weights = torch.tensor(CLASS_WEIGHTS).to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    train_df, valid_df = load_data()

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["statement"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="models/roberta_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )

    mlflow.set_experiment("fake_news_detection")

    with mlflow.start_run(run_name="RoBERTa Weighted Loss"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("class_weights", str(CLASS_WEIGHTS))

        trainer.train()

        metrics = trainer.evaluate()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        os.makedirs("models/final_roberta_fake_news", exist_ok=True)

        model.save_pretrained("models/final_roberta_fake_news")
        tokenizer.save_pretrained("models/final_roberta_fake_news")

        mlflow.log_artifacts(
            "models/final_roberta_fake_news",
            artifact_path="final_roberta_fake_news"
        )

        print(metrics)
        print("Weighted RoBERTa training completed and logged to MLflow.")


if __name__ == "__main__":
    main()