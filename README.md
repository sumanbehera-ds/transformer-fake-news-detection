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



