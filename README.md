🏦 Loan Default Prediction API with Explainable AI (XAI)


A production-ready RESTful API for loan default prediction with SHAP-based explainability, containerized with Docker and deployed on Render.

🌟 Live Demo
🔗 API Documentation: https://loan-default-prediction-xai.onrender.com/docs

✨ Features
✅ Machine Learning Model: XGBoost classifier for loan default prediction

✅ Explainable AI (XAI): SHAP values for model interpretability

✅ RESTful API: FastAPI with auto-generated interactive documentation

✅ Batch Predictions: Process multiple loan applications simultaneously

✅ Performance Monitoring: Real-time API metrics and health checks

✅ Dockerized: Containerized application for consistent deployments

✅ Cloud Deployed: Production-ready API hosted on Render

🛠️ Tech Stack
Machine Learning
XGBoost, Scikit-learn, SHAP, Pandas, NumPy

Backend
FastAPI, Pydantic, Uvicorn

DevOps
Docker, GitHub, Render

🏗️ Project Architecture
text
loan-default-prediction-xai/
├── api/
│   ├── app.py              # FastAPI application
│   ├── schemas.py          # Pydantic models
│   └── monitoring.py       # Performance tracking
├── models/
│   ├── xgboost_model.joblib
│   └── preprocessor.joblib
├── src/
│   ├── train.py
│   ├── preprocessing.py
│   └── explainability.py
├── Dockerfile
└── requirements.txt





🔌 API Endpoints
1. Health Check
text
GET /health
Response:

json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
2. Single Prediction
text
POST /predict
Request:

json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 75000,
  "CoapplicantIncome": 0,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
Response:

json
{
  "prediction": "Approved",
  "probability": 0.8261,
  "top_factors": [
    {
      "feature": "LoanAmount",
      "contribution": 0.6364,
      "impact": "Positive"
    },
    {
      "feature": "Credit_History",
      "contribution": 0.5379,
      "impact": "Positive"
    }
  ]
}
3. Batch Prediction
text
POST /batch_predict
4. Performance Metrics
text
GET /metrics
5. Model Information
text
GET /model_info
🚀 Installation & Usage
Local Setup
bash
# Clone repository
git clone https://github.com/PrasanthKumarS777/loan-default-prediction-xai.git
cd loan-default-prediction-xai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
Access at: http://localhost:8000/docs

Docker Deployment
bash
# Build image
docker build -t loan-api .

# Run container
docker run -d -p 8000:8000 --name loan-container loan-api
💡 Usage Example
Python
python
import requests

url = "https://loan-default-prediction-xai.onrender.com/predict"
data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 75000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}

response = requests.post(url, json=data)
print(response.json())
📊 Model Performance
Algorithm: XGBoost Classifier

Features: 14 engineered features

Accuracy: ~85%

Key Predictors: Credit History, Loan Amount, Applicant Income

🔍 Explainability (XAI)
SHAP (SHapley Additive exPlanations) provides:

Feature importance rankings

Individual prediction explanations

Positive/negative impact analysis

🌐 Cloud Deployment
Deployed on Render with CI/CD:

Automatic deployments from GitHub

Zero-downtime updates

Docker-based deployment

Live URL: https://loan-default-prediction-xai.onrender.com/docs

👤 Author
Prasanth Kumar Sahu

GitHub: @PrasanthKumarS777

Project: loan-default-prediction-xai

🙏 Acknowledgments
FastAPI for the web framework

SHAP for explainability

Render for hosting
