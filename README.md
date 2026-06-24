# 📰 Transformer Fake News Detection

A production-ready Natural Language Processing (NLP) system for detecting fake news using Transformer models. This project compares traditional machine learning with modern Transformer architectures and deploys the best-performing model using FastAPI, Docker, MLflow, and Hugging Face.

---

## 🚀 Overview

Fake news spreads rapidly across digital platforms and can significantly influence public opinion. This project builds an end-to-end fake news detection pipeline that classifies short news claims as **REAL** or **FAKE** using multiple machine learning and deep learning approaches.

The project includes:

* Data preprocessing and exploratory analysis
* TF-IDF baseline models
* Transformer fine-tuning (DistilBERT & RoBERTa)
* Experiment tracking with MLflow
* REST API using FastAPI
* Docker containerization
* Interactive Hugging Face deployment

---

# 📊 Dataset

**Dataset:** LIAR Dataset

The LIAR dataset contains short political statements labeled with six truthfulness categories.

These labels were converted into a binary classification problem:

| Original Labels                | Binary Label |
| ------------------------------ | ------------ |
| pants-fire, false, barely-true | FAKE         |
| half-true, mostly-true, true   | REAL         |

After preprocessing:

* REAL: **3,638**
* FAKE: **6,602**

---

# 🧠 Models Evaluated

| Model                        |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ---------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression |     0.6768 |     0.5118 |     0.2571 |     0.3423 |     0.6702 |
| TF-IDF + Naive Bayes         |     0.6651 |     0.4490 |     0.1048 |     0.1699 |     0.6565 |
| DistilBERT                   |     0.6900 |     0.5327 |     0.4261 |     0.4735 |          — |
| RoBERTa                      |     0.6916 |     0.5397 |     0.3881 |     0.4515 |          — |
| **Weighted RoBERTa (Best)**  | **0.6768** | **0.5047** | **0.6452** | **0.5664** | **0.7152** |

---

# ⭐ Key Results

* Improved F1 Score from **0.3423 → 0.5664** using Transformer fine-tuning.
* Increased Recall from **25.7% → 64.5%**, reducing missed fake news.
* Compared classical ML and Transformer-based architectures.
* Tracked experiments with MLflow.
* Built a production-ready REST API using FastAPI.
* Dockerized the complete application.
* Deployed an interactive inference application on Hugging Face.

---

# 🛠 Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* PyTorch
* Hugging Face Transformers

### NLP

* TF-IDF
* Logistic Regression
* Naive Bayes
* DistilBERT
* RoBERTa

### Deployment

* FastAPI
* Docker
* Hugging Face Spaces

### MLOps

* MLflow
* Git
* GitHub

---

# 📁 Project Structure

```text
transformer-fake-news-detection/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── notebooks/
├── models/
├── reports/
├── streamlit/
└── screenshots/
```

---

# 🚀 Installation

```bash
git clone https://github.com/sumanbehera-ds/transformer-fake-news-detection.git

cd transformer-fake-news-detection

pip install -r requirements.txt
```

---

# ▶️ Run FastAPI

```bash
uvicorn app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker

Build

```bash
docker build -t fake-news-detector .
```

Run

```bash
docker run -p 8000:8000 fake-news-detector
```

---

# 🌐 Live Demo

## Hugging Face Space

https://huggingface.co/spaces/sumanbehera-ds/roberta-fake-news-api

## Hugging Face Model

https://huggingface.co/sumanbehera-ds/roberta-fake-news-detector

---

# 📈 Future Improvements

* Explain predictions using SHAP or LIME
* Add batch prediction endpoint
* Integrate real-time news verification APIs
* Add CI/CD with GitHub Actions
* Continuous model monitoring
* Automated retraining pipeline

---

# ⚠️ Limitations

This model is trained on the LIAR dataset, which contains short political claims. It identifies linguistic patterns learned during training and should not be considered a real-time fact-checking system or a source of verified truth.

---
# 👨‍💻 Author

**Suman Behera**

AI/ML Engineer | Data Scientist

GitHub:
https://github.com/sumanbehera-ds

LinkedIn:
https://linkedin.com/in/suman-01-behera


