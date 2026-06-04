import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os

COLS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts",
    "mostly_true_counts", "pants_fire_counts", "context"
]

def convert_label(label):
    return 1 if label in ["true", "mostly-true"] else 0

train_df = pd.read_csv("data/raw/train.tsv", sep="\t", header=None, names=COLS)
valid_df = pd.read_csv("data/raw/valid.tsv", sep="\t", header=None, names=COLS)

X_train = train_df["statement"]
y_train = train_df["label"].apply(convert_label)

X_valid = valid_df["statement"]
y_valid = valid_df["label"].apply(convert_label)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

mlflow.set_experiment("fake_news_detection")

with mlflow.start_run(run_name="TF-IDF Logistic Regression Auto"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred),
        "roc_auc": roc_auc_score(y_valid, y_prob)
    }

    mlflow.log_param("model", "TF-IDF + Logistic Regression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("max_iter", 1000)

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    os.makedirs("models", exist_ok=True)

    with open("models/tfidf_logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, "model")

    print(metrics)
    print("Training completed and logged to MLflow.")