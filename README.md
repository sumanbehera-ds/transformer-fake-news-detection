# 📰 Transformer Fake News Detection

An end-to-end NLP system for detecting fake news using Transformer models. This project benchmarks classical ML baselines against fine-tuned Transformer architectures and deploys the best-performing model via FastAPI, Docker, MLflow, and Hugging Face.

---

## 🌐 Live Demo

| | Link |
|---|---|
| **Hugging Face Space** | https://huggingface.co/spaces/sumanbehera-ds/roberta-fake-news-api |
| **Hugging Face Model** | https://huggingface.co/sumanbehera-ds/roberta-fake-news-detector |

---

## ⭐ Key Results

| Metric | Baseline (TF-IDF + LR) | Best Model (Weighted RoBERTa) | Improvement |
|---|---|---|---|
| F1 Score | 0.3423 | **0.5740** | **+67.7%** |
| Recall | 25.7% | **68.3%** | **+42.6pp** |
| ROC-AUC | 0.6702 | **0.7208** | **+7.5%** |

> **Metric provenance:** The full comparison table and deployed model/API links are recorded in [`reports/metrics.md`](reports/metrics.md). Notebook 2 includes executed Colab outputs for the weighted RoBERTa training and evaluation run.

---

## 📊 Dataset

**LIAR Dataset** — short political statements labeled with six truthfulness categories, converted to binary classification:

| Original Labels | Binary Label |
|---|---|
| pants-fire, false, barely-true, half-true | FAKE (0) |
| mostly-true, true | REAL (1) |

| Split | Samples |
|---|---|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |

Class distribution: FAKE **6,602** · REAL **3,638** (imbalanced — addressed via weighted loss)

---

## 🧠 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.6768 | 0.5118 | 0.2571 | 0.3423 | 0.6702 |
| TF-IDF + Naive Bayes | 0.6651 | 0.4490 | 0.1048 | 0.1699 | 0.6565 |
| DistilBERT | 0.6900 | 0.5327 | 0.4261 | 0.4735 | — |
| RoBERTa (standard) | 0.6869 | 0.5205 | 0.5429 | 0.5315 | 0.7105 |
| **Weighted RoBERTa (Best)** | **0.6682** | **0.4948** | **0.6833** | **0.5740** | **0.7208** |

> **Why Weighted RoBERTa?** Standard RoBERTa optimizes accuracy but underperforms on the minority class (REAL). A custom `WeightedTrainer` with `CrossEntropyLoss(weight=[1.0, 1.8])` significantly improved Recall and F1 on the imbalanced LIAR dataset.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / DL | Scikit-learn, PyTorch, Hugging Face Transformers |
| NLP | TF-IDF, DistilBERT, RoBERTa |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
transformer-fake-news-detection/
│
├── app.py                        # FastAPI REST API (calls HF Inference API)
├── Dockerfile                    # Docker config for API deployment
├── requirements.txt              # API dependencies
├── requirements-train.txt        # Full training dependencies
│
├── src/
│   └── models/
│       ├── train_model.py        # TF-IDF baseline training + MLflow logging
│       ├── train_roberta.py      # Weighted RoBERTa fine-tuning + MLflow logging
│       ├── mlflow_tracking.py    # Backfill script for MLflow experiment runs
│       └── test_model.py         # Local model inference test
│
├── notebooks/
│   ├── 01_eda_preprocessing_baseline.ipynb   # EDA + baseline model
│   └── 02_transformer_training_colab.ipynb   # Transformer training (Colab)
│
├── models/
│   ├── tfidf_logistic_model.pkl              # Trained baseline model
│   └── final_roberta_fake_news/              # RoBERTa tokenizer + config
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
└── streamlit_app/
    └── app.py                    # Streamlit UI (runs alongside FastAPI locally)
```

---

## 🚀 Installation

```bash
git clone https://github.com/sumanbehera-ds/transformer-fake-news-detection.git
cd transformer-fake-news-detection
```

The committed `models/final_roberta_fake_news/` folder contains tokenizer/config files only. Large model weight files are not tracked; `src/models/test_model.py` falls back to the deployed Hugging Face model when local weights are absent.

**For API only:**
```bash
pip install -r requirements.txt
```

**For Streamlit UI:**
```bash
pip install -r requirements-ui.txt
```

**For training:**
```bash
pip install -r requirements-train.txt
```

---

## ▶️ Run FastAPI

Set your Hugging Face token first:
```bash
export HF_TOKEN=your_huggingface_token
```

Then start the API:
```bash
uvicorn app:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

**API Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | API status + model ID |
| `/debug` | GET | Token and config info |
| `/predict` | POST | Classify a news statement |

**Example request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The government secretly controls the weather."}'
```

**Example response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.8923,
  "raw_output": [
    {"label": "LABEL_0", "score": 0.8923},
    {"label": "LABEL_1", "score": 0.1077}
  ]
}
```

---

## 🖥️ Run Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

By default, the UI calls `http://127.0.0.1:8000/predict`. Override it when needed:

```bash
API_URL=http://127.0.0.1:7860/predict streamlit run streamlit_app/app.py
```

---

## 🐳 Docker

**Build:**
```bash
docker build -t fake-news-detector .
```

**Run:**
```bash
docker run -p 7860:7860 -e HF_TOKEN=your_token fake-news-detector
```

---

## 🧪 Training

Before running training scripts, download the LIAR dataset and place the split files at:

```text
data/raw/train.tsv
data/raw/valid.tsv
data/raw/test.tsv
```

The `data/` directory is intentionally ignored to keep the repository lightweight.

**Train baseline (TF-IDF + Logistic Regression):**
```bash
python src/models/train_model.py
```

**Fine-tune Weighted RoBERTa:**
```bash
python src/models/train_roberta.py
```

All runs are tracked in MLflow under the `fake_news_detection` experiment.

```bash
mlflow ui
```

Open: `http://127.0.0.1:5000`

---

## ⚠️ Limitations

- Trained on the LIAR dataset which contains short political claims (avg. 107 chars). Performance on long-form news articles may differ.
- Binary label mapping collapses six nuanced categories — `half-true` is classified as FAKE in this implementation.
- Should not be used as a real-time fact-checking system or source of verified truth.

---

## 📈 Future Improvements

- Explainability with SHAP or LIME
- Batch prediction endpoint
- CI/CD with GitHub Actions
- Real-time news API integration
- Continuous model monitoring and automated retraining

---

## 👨‍💻 Author

**Suman Behera** · AI/ML Engineer | Data Scientist

[![GitHub](https://img.shields.io/badge/GitHub-sumanbehera--ds-181717?logo=github)](https://github.com/sumanbehera-ds)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-suman--01--behera-0077B5?logo=linkedin)](https://linkedin.com/in/suman-01-behera)
