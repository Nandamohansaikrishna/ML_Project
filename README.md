# 💳 Credit Risk Scorer & AI Advisor

An end-to-end Machine Learning system that predicts credit card default risk, explains decisions using **SHAP values**, and provides loan advice via an interactive **Groq LLM** chat interface.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)

---

## 📌 Features

- **Default Prediction** — Evaluates applicant credit default risk with calibrated probability scores.
- **Model Explainability** — Generates local feature-importance attributions using TreeSHAP.
- **AI Loan Advisor** — Chat interface powered by Groq LLM to deliver automated risk assessments.
- **Dual-View Dashboard** — Streamlit UI with synchronized customer records, benchmark targets, and dark/light modes.
- **Production REST API** — High-throughput FastAPI endpoints with automated Swagger documentation.

---

## 🏗️ Architecture Pipeline

```
Raw Data (30K Records)
├── 1. EDA & Cleaning          # Duplicate removal, categorical remapping
├── 2. Feature Engineering     # 29 features, outlier capping, SMOTE on train split
├── 3. Model Training          # Random Forest vs. Baseline Logistic Regression
├── 4. Explainability          # SHAP TreeExplainer + Groq LLM API
└── 5. Deployment              # FastAPI backend + Streamlit frontend
```

---

## 📊 Model Performance

Evaluated on a 15% held-out test set (UCI Credit Card Dataset):

| Model | AUC-ROC | F1-Score | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression *(Baseline)* | 0.718 | 0.466 | 0.379 | **0.603** |
| **Random Forest *(Production)*** | **0.772** | **0.529** | **0.490** | 0.576 |

> **Key Finding:** Recent payment delay status (`PAY_0`) and credit utilization rate are the strongest predictors of default.

---

## 🚀 Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Nandamohansaikrishna/ML_Project.git
cd credit-risk-scorer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Application

Start both services with the unified runner:

```bash
python run.py
```

- **Web UI:** http://localhost:8501
- **FastAPI Docs:** http://localhost:8000/docs

Or launch them separately:

```bash
# Terminal 1: Backend
uvicorn app:app --reload --port 8000

# Terminal 2: Dashboard
streamlit run streamlit_app.py
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/` | Health check |
| `GET` | `/model-info` | Model metadata and evaluation metrics |
| `POST` | `/predict` | Returns risk prediction and default probability |
| `POST` | `/explain` | Returns risk prediction, SHAP attributions, and Groq LLM advice |

---

## 📂 Project Structure

```
credit-risk-scorer/
├── notebooks/           # Phase 1-4 Jupyter notebooks (EDA to SHAP agent)
├── src/                 # Reusable utility scripts and inference pipelines
├── models/              # Pickled estimators and fitted transformers
├── data/                # Processed splits and display data
├── app.py               # FastAPI application
├── streamlit_app.py     # Streamlit web interface
├── run.py               # Concurrent server launcher
└── requirements.txt     # Python project dependencies
```
