<div align="center">

# 🛡️ AI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Deployed-Cloudflare%20Workers-F38020?style=flat&logo=cloudflare&logoColor=white)

*Real-time transaction fraud detection with ML pipeline, REST API, and Cloudflare Workers deployment*

</div>

---

## ✨ Features

- Real-time fraud prediction with <100ms latency
- Gradient Boosting model with 98%+ accuracy
- Automated feature engineering pipeline
- RESTful API with OpenAPI documentation
- PostgreSQL for transaction storage
- Docker containerization
- Cloudflare Workers edge deployment
- Comprehensive test suite

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24-2496ED?style=flat&logo=docker&logoColor=white)
![Cloudflare](https://img.shields.io/badge/Cloudflare-Workers-F38020?style=flat&logo=cloudflare&logoColor=white)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Raphasha27/python-ai-fraud-detection.git
cd python-ai-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Setup database
docker-compose up -d postgres

# Train model
python scripts/train_model.py

# Run API
uvicorn src.main:app --reload --port 8000
```

### Docker Deployment

```bash
docker-compose up --build
```

### Cloudflare Workers

```bash
npm install -g wrangler
wrangler deploy
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict fraud for transaction |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Model metrics |
| `GET` | `/transactions` | List transactions |
| `GET` | `/transactions/{id}` | Get transaction details |

## 🏗️ Architecture

```
Transaction Request
        │
        ▼
┌─────────────────┐
│   FastAPI       │
│   (Cloudflare)  │
└────────┬────────┘
         │
┌────────▼────────┐
│  Feature        │
│  Engineering    │
└────────┬────────┘
         │
┌────────▼────────┐
│  ML Model       │
│  (XGBoost)      │
└────────┬────────┘
         │
┌────────▼────────┐
│  PostgreSQL     │
│  (Storage)      │
└─────────────────┘
```

## 🌐 Live Demo

| Platform | URL |
|----------|-----|
| GitHub Pages | [raphasha27.github.io/python-ai-fraud-detection](https://raphasha27.github.io/python-ai-fraud-detection) |
| Docker Hub | [hub.docker.com/r/raphasha27/python-ai-fraud-detection](https://hub.docker.com/r/raphasha27/python-ai-fraud-detection) |
| Cloudflare Pages | [fraud-detection-api-3pg.pages.dev](https://fraud-detection-api-3pg.pages.dev) |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.2% |
| Precision | 96.8% |
| Recall | 94.5% |
| F1 Score | 95.6% |
| AUC-ROC | 0.991 |

## 👤 Author

**raphasha27** — [GitHub](https://github.com/raphasha27)
