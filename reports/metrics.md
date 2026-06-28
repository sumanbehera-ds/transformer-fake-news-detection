# Fake News Detection Metrics

This file records the project metrics used in the README and notebook narrative.

## Verification Links

- Final deployed model: https://huggingface.co/sumanbehera-ds/roberta-fake-news-detector
- Live API Space: https://huggingface.co/spaces/sumanbehera-ds/roberta-fake-news-api

## Validation Metrics

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6768 | 0.5118 | 0.2571 | 0.3423 | 0.6702 |
| TF-IDF + Naive Bayes | 0.6651 | 0.4490 | 0.1048 | 0.1699 | 0.6565 |
| DistilBERT | 0.6900 | 0.5327 | 0.4261 | 0.4735 | N/A |
| RoBERTa | 0.6869 | 0.5205 | 0.5429 | 0.5315 | 0.7105 |
| Weighted RoBERTa | 0.6682 | 0.4948 | 0.6833 | 0.5740 | 0.7208 |

## Reproduced Colab Metrics

These values come from the executed weighted RoBERTa cells in `notebooks/02_transformer_training_colab.ipynb`.

| Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.6682 | 0.4948 | 0.6833 | 0.5740 | 0.7208 |
| Test | 0.6156 | 0.4694 | 0.6481 | 0.5444 | 0.6840 |

## Reproducibility Note

The transformer training notebook was run in Google Colab with a GPU runtime. The checked-in notebook includes the executed weighted RoBERTa training, validation, and test outputs.

For maximum auditability, re-run `notebooks/02_transformer_training_colab.ipynb` in Colab with a GPU runtime and commit the notebook with outputs preserved.
