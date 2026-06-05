# Fake News Detection using Transformers (DistilBERT & RoBERTa)

**Author:** Suman Behera

## Overview

This project builds an end-to-end Fake News Detection system using traditional Machine Learning and Transformer-based NLP models on the LIAR dataset. The project follows a production-oriented workflow including data preprocessing, model experimentation, MLflow tracking, API development, containerization, and cloud deployment.

The objective is to classify news statements as **REAL** or **FAKE** while comparing classical ML approaches with modern Transformer architectures.

---

## Features

* Data preprocessing and label engineering
* Baseline ML models

  * TF-IDF + Logistic Regression
  * Multinomial Naive Bayes
* Transformer models

  * DistilBERT
  * RoBERTa
  * Weighted RoBERTa (Final Model)
* MLflow experiment tracking
* FastAPI inference API
* Docker containerization
* Hugging Face Model Hub integration
* Hugging Face Spaces deployment

---

## Dataset

**Dataset:** LIAR Dataset

The LIAR dataset contains short political statements labeled with truthfulness ratings.

For binary classification:

| Original Labels      | Mapped Label |
| -------------------- | ------------ |
| true, mostly-true    | REAL (1)     |
| all remaining labels | FAKE (0)     |

---

## Model Performance

| Model                              |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression       |     0.6768 |     0.5118 |     0.2571 |     0.3423 |     0.6702 |
| Multinomial Naive Bayes            |     0.6651 |     0.4490 |     0.1048 |     0.1699 |     0.6565 |
| DistilBERT                         |     0.6900 |     0.5327 |     0.4261 |     0.4735 |          - |
| RoBERTa                            |     0.6869 |     0.5205 |     0.5429 |     0.5315 |     0.7105 |
| **Weighted RoBERTa (Final Model)** | **0.6768** | **0.5047** | **0.6452** | **0.5664** | **0.7152** |

### Final Production Model

Weighted RoBERTa achieved the best overall performance by improving recall and F1 score through class-weighted loss during training.

---

## Project Architecture

```text
LIAR Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
 ├── Logistic Regression
 ├── Naive Bayes
 ├── DistilBERT
 └── RoBERTa
      │
      ▼
MLflow Tracking
      │
      ▼
Best Model Selection
      │
      ▼
Weighted RoBERTa
      │
      ▼
FastAPI
      │
      ▼
Docker
      │
      ▼
Hugging Face Spaces
```

## Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

### NLP & Deep Learning

* Transformers
* PyTorch
* Hugging Face

### MLOps

* MLflow
* FastAPI
* Docker
* Hugging Face Model Hub
* Hugging Face Spaces

---

## API Endpoints

### Home

```http
GET /
```

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Example Request:

```json
{
  "text": "The government secretly controls the weather using hidden satellites."
}
```

Example Response:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94
}
```

---

## Running Locally

### Clone Repository

```bash
git clone https://github.com/sumanbehera-ds/transformer-fake-news-detection.git
cd transformer-fake-news-detection
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Start API

```bash
uvicorn app:app --reload
```

---

## Docker

Build Docker Image:

```bash
docker build -t fake-news-api .
```

Run Container:

```bash
docker run -p 8000:8000 fake-news-api
```

---

## MLflow Tracking

Launch MLflow UI:

```bash
python -m mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Tracked Experiments:

* TF-IDF Logistic Regression
* Multinomial Naive Bayes
* DistilBERT
* RoBERTa
* Weighted RoBERTa

---

## Deployment

### Hugging Face Model Hub

Stores the final production model artifacts:

* config.json
* model.safetensors
* tokenizer.json
* tokenizer_config.json

### Hugging Face Spaces

Hosts the live FastAPI application for inference.

---

## Future Improvements

* Hyperparameter optimization
* Metadata-aware transformer architecture
* Threshold optimization
* Model monitoring and drift detection
* CI/CD automation

---

## Author

**Suman Behera**

Aspiring Data Scientist focused on Machine Learning, NLP, Deep Learning, and MLOps.

GitHub: https://github.com/sumanbehera-ds
