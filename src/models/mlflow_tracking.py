import mlflow

mlflow.set_experiment("fake_news_detection")

all_models = [
    {
        "run_name": "TF-IDF Logistic Regression",
        "model_type": "Baseline ML",
        "algorithm": "Logistic Regression",
        "accuracy": 0.6768,
        "precision": 0.5118,
        "recall": 0.2571,
        "f1": 0.3423,
        "roc_auc": 0.6702,
    },
    {
        "run_name": "TF-IDF Naive Bayes",
        "model_type": "Baseline ML",
        "algorithm": "Multinomial Naive Bayes",
        "accuracy": 0.6651,
        "precision": 0.4490,
        "recall": 0.1048,
        "f1": 0.1699,
        "roc_auc": 0.6565,
    },
    {
        "run_name": "DistilBERT",
        "model_type": "Transformer",
        "algorithm": "DistilBERT",
        "accuracy": 0.6900,
        "precision": 0.5327,
        "recall": 0.4261,
        "f1": 0.4735,
        "roc_auc": None,
    },
    {
        "run_name": "RoBERTa Colab Auto Training",
        "model_type": "Transformer",
        "algorithm": "RoBERTa",
        "accuracy": 0.6869,
        "precision": 0.5205,
        "recall": 0.5429,
        "f1": 0.5315,
        "roc_auc": 0.7105,
    },
]

for model in all_models:
    with mlflow.start_run(run_name=model["run_name"]):
        mlflow.log_param("model_type", model["model_type"])
        mlflow.log_param("algorithm", model["algorithm"])
        mlflow.log_param("dataset", "LIAR")
        mlflow.log_param("task", "Binary Fake News Classification")
        mlflow.log_param("labels", "FAKE=0, REAL=1")

        mlflow.log_metric("accuracy", model["accuracy"])
        mlflow.log_metric("precision", model["precision"])
        mlflow.log_metric("recall", model["recall"])
        mlflow.log_metric("f1", model["f1"])

        if model["roc_auc"] is not None:
            mlflow.log_metric("roc_auc", model["roc_auc"])

        print(f"Logged: {model['run_name']}")

print("All model runs logged successfully.")