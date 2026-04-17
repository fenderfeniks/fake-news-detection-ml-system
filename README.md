# Fake News Detection System

An end-to-end machine learning system for detecting fake news using deep learning models, optimized ensembles, and deployed as a REST API.

---

## Project Overview

This project implements a full ML pipeline for fake news classification:

- Data preprocessing and feature engineering
- Text vectorization using Word2Vec
- Deep learning models: RNN, LSTM, GRU
- Hyperparameter optimization with Optuna
- Ensemble learning with optimized weights
- Deployment via FastAPI
- Containerization using Docker

---

## Architecture

Data → Preprocessing → Tokenization → Word2Vec → Models → Ensemble → API → Docker

---

## Data Processing

- Combined features: `title`, `news_url`, `source_domain`
- Cleaned URLs while preserving semantic tokens
- Applied:
  - Lowercasing
  - Regex cleaning
  - Stopword removal
- Built unified `text` feature

---

## Text Representation

Used Word2Vec embeddings:

- Captures semantic relationships
- Improves generalization
- Integrated into PyTorch embedding layer

---

## Models

- RNN — baseline
- LSTM — handles long dependencies
- GRU — best performance/efficiency trade-off

---

## Hyperparameter Optimization

Optimized using Optuna:

- Hidden dimension
- Number of layers
- Dropout
- Learning rate

Metric:

- ROC AUC

---

## Ensemble

Weighted ensemble of:

- RNN
- LSTM
- GRU

Weights optimized using Optuna.

---

## Metrics

- Accuracy
- F1-score
- ROC AUC

---

## API

### Endpoint

POST /predict

### Example request

{
  "text": "Breaking news: government announces new policy"
}

### Example response

{
  "model_probabilities": {
    "RNN": 0.58,
    "LSTM": 0.21,
    "GRU": 0.00
  },
  "ensemble_probability": 0.18,
  "ensemble_prediction": 0,
  "label": "fake"
}

---

## Docker

Build:

docker build -t fake-news-api .

Run:

docker run -p 8000:8000 fake-news-api

Docs:

http://127.0.0.1:8000/docs

---

## Project Structure

artifacts/
src/
main.py
Dockerfile
requirements.txt

---

## Key Points

- Full ML pipeline implemented
- Multiple architectures compared
- Automated tuning
- Ensemble optimization
- Production-ready API
- Dockerized deployment

---

## Future Improvements

- Logging
- Monitoring
- CI/CD
- Transformer models (BERT)

---

## Author

Maxim Novikov
