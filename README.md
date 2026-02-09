# 🏦 Loan Default Prediction API with Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A production-ready machine learning system for predicting loan defaults with transparent, explainable AI powered by SHAP values.**

[🚀 Live Demo](https://loan-default-prediction-xai.vercel.app) · [📚 API Docs](https://loan-default-prediction-xai.onrender.com/docs) · [🐛 Report Bug](https://github.com/PrasanthKumarS777/loan-default-prediction-xai/issues) · [✨ Request Feature](https://github.com/PrasanthKumarS777/loan-default-prediction-xai/issues)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Performance](#-model-performance)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting loan defaults using **XGBoost** with **SHAP-based explainability**. The system features a modern 3D React frontend and a high-performance FastAPI backend, designed for real-world production deployment.

**Key Capabilities:**
- Real-time loan approval predictions with <100ms latency
- Transparent AI decision-making through SHAP feature importance
- 93.2% accuracy with production-grade reliability
- Scalable cloud architecture on Render and Vercel

---

## 🌐 Live Demo

| Component | URL | Description |
|-----------|-----|-------------|
| **🎨 Frontend** | [loan-default-prediction-xai.vercel.app](https://loan-default-prediction-xai.vercel.app) | Interactive React UI with 3D animations |
| **🚀 Backend API** | [loan-default-prediction-xai.onrender.com](https://loan-default-prediction-xai.onrender.com) | RESTful API server |
| **📚 API Docs** | [Swagger UI](https://loan-default-prediction-xai.onrender.com/docs) | Interactive API documentation |

---

## ✨ Features

### 🤖 Machine Learning
- **XGBoost Classifier** - State-of-the-art gradient boosting for binary classification
- **SHAP Explainability** - Understand every prediction with Shapley values
- **93.2% Accuracy** - Rigorously validated on holdout test data
- **Real-time Inference** - Sub-100ms prediction latency
- **Feature Engineering** - Advanced domain-driven transformations

### 🔧 Backend (FastAPI)
- **RESTful API** - Clean, well-documented endpoints
- **Single & Batch Predictions** - Flexible processing modes
- **Health Monitoring** - Built-in status checks and metrics
- **CORS Support** - Cross-origin resource sharing enabled
- **Input Validation** - Pydantic schema validation
- **Error Handling** - Comprehensive exception management
- **Docker Ready** - Containerized for consistent deployment

### 🎨 Frontend (React + Vite)
- **3D Particle Effects** - Animated background with interactive particles
- **Glassmorphism UI** - Modern frosted glass design aesthetic
- **Interactive Charts** - Radar plots, bar charts, and gauges (Recharts)
- **Live Statistics** - Real-time system health dashboard
- **Risk Visualization** - Dynamic risk score meter with color gradients
- **Responsive Design** - Mobile-first, works on all screen sizes
- **Dark Theme** - Professional grey and red color scheme
- **Smooth Animations** - CSS transitions and loading states

### ⚙️ DevOps & Infrastructure
- **CI/CD Pipeline** - Automated testing via GitHub Actions
- **Cloud Deployment** - Backend on Render, Frontend on Vercel
- **Version Control** - Professional Git workflow
- **Environment Management** - Secure configuration with `.env`
- **Monitoring** - Health checks and performance metrics

---

## 🛠️ Tech Stack

<table>
<tr>
<td width="33%" valign="top">

### 🔬 Data Science & ML
- **Python 3.11** - Core language
- **XGBoost** - Gradient boosting
- **Scikit-learn** - ML utilities
- **SHAP** - Model explainability
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

</td>
<td width="33%" valign="top">

### 🖥️ Backend
- **FastAPI** - Async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Joblib** - Model serialization
- **Python-dotenv** - Environment management

</td>
<td width="33%" valign="top">

### 🎨 Frontend
- **React 19** - UI library
- **Vite** - Build tool
- **Recharts** - Data visualization
- **Lucide React** - Icon library
- **Axios** - HTTP client
- **CSS3** - Styling & animations

</td>
</tr>
<tr>
<td colspan="3" align="center">

### ☁️ DevOps & Deployment
**Docker** • **GitHub Actions** • **Render** • **Vercel** • **Git**

</td>
</tr>
</table>

---

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 93.2% | Overall correct predictions |
| **Precision** | 91.5% | True positive rate |
| **Recall** | 89.8% | Sensitivity (TPR) |
| **F1-Score** | 90.6% | Harmonic mean of precision & recall |
| **ROC-AUC** | 95.3% | Area under ROC curve |

### 🔍 SHAP Explainability

The model provides transparent predictions using SHAP (SHapley Additive exPlanations):

- **Feature Importance** - Identify top factors influencing each decision
- **Contribution Values** - Quantify positive/negative impact of each feature
- **Transparent Decisions** - Clear reasoning for loan approvals or rejections
- **Regulatory Compliance** - Explain model decisions to stakeholders

---

## 📦 Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **npm** or **yarn**
- **Git** ([Download](https://git-scm.com/))

### 1️⃣ Clone Repository

```bash
git clone https://github.com/PrasanthKumarS777/loan-default-prediction-xai.git
cd loan-default-prediction-xai
```

### 2️⃣ Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

✅ **Backend running at:** `http://localhost:8000`  
📚 **API docs available at:** `http://localhost:8000/docs`

### 3️⃣ Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

✅ **Frontend running at:** `http://localhost:5173`

### 4️⃣ Docker Setup (Optional)

```bash
# Build Docker image
docker build -t loan-default-api .

# Run container
docker run -p 8000:8000 loan-default-api

# Access API at http://localhost:8000
```

---

## 🚀 API Documentation

### Base URL
- **Local:** `http://localhost:8000`
- **Production:** `https://loan-default-prediction-xai.onrender.com`

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-09T12:00:00Z"
}
```

#### 2. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "person_age": 25,
  "person_income": 45000,
  "person_emp_length": 3,
  "loan_amnt": 10000,
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.22,
  "cb_person_cred_hist_length": 5,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "cb_person_default_on_file": "N"
}
```

**Response:**
```json
{
  "prediction": "Approved",
  "probability": 0.8542,
  "risk_score": 15.8,
  "top_factors": [
    {
      "feature": "Credit History Length",
      "contribution": 0.2345,
      "impact": "Positive"
    },
    {
      "feature": "Loan to Income Ratio",
      "contribution": -0.1234,
      "impact": "Negative"
    }
  ]
}
```

#### 3. Batch Prediction
```http
POST /predict/batch
Content-Type: application/json
```

**Request:** Array of loan application objects  
**Response:** Array of prediction results

#### 4. Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "XGBoost",
  "version": "2.0",
  "accuracy": 0.932,
  "features": 11,
  "trained_on": "2025-01-15"
}
```

#### 5. Performance Metrics
```http
GET /metrics
```

---

## 📁 Project Structure

```
loan-default-prediction-xai/
│
├── 📂 api/                         # FastAPI Backend
│   ├── main.py                     # API application & routes
│   ├── schemas.py                  # Pydantic data models
│   └── __init__.py
│
├── 📂 frontend/                    # React Frontend
│   ├── 📂 src/
│   │   ├── App.jsx                # Main React component
│   │   ├── App.css                # Styles with 3D animations
│   │   ├── main.jsx               # Entry point
│   │   └── assets/                # Images, icons, etc.
│   ├── 📂 public/                 # Static assets
│   ├── index.html                 # HTML template
│   ├── package.json               # Node dependencies
│   └── vite.config.js             # Vite configuration
│
├── 📂 src/                         # ML Source Code
│   ├── data_preprocessing.py      # Data cleaning & transformation
│   ├── feature_engineering.py     # Feature creation
│   ├── model_training.py          # XGBoost training pipeline
│   ├── explainability.py          # SHAP integration
│   └── utils.py                   # Helper functions
│
├── 📂 models/                      # Trained Models
│   ├── loan_model.pkl             # XGBoost classifier
│   ├── preprocessor.pkl           # Scikit-learn preprocessor
│   └── feature_names.json         # Feature metadata
│
├── 📂 notebooks/                   # Jupyter Notebooks
│   ├── EDA.ipynb                  # Exploratory data analysis
│   ├── model_development.ipynb    # Model experimentation
│   └── shap_analysis.ipynb        # Explainability research
│
├── 📂 tests/                       # Unit Tests
│   ├── test_api.py                # API endpoint tests
│   ├── test_model.py              # Model prediction tests
│   └── test_preprocessing.py      # Data pipeline tests
│
├── 📂 .github/                     # GitHub Configuration
│   └── workflows/
│       └── ci-cd.yml              # CI/CD pipeline
│
├── 📄 Dockerfile                   # Docker configuration
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # This file
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# Model Paths
MODEL_PATH=models/loan_model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl

# Frontend Configuration (frontend/.env)
VITE_API_URL=https://loan-default-prediction-xai.onrender.com

# Optional: Monitoring
LOG_LEVEL=INFO
```

⚠️ **Never commit `.env` files to version control!**

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=src --cov=api tests/

# Run specific test file
pytest tests/test_api.py -v

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

### Test Coverage Goals
- **API Endpoints:** >90%
- **Model Pipeline:** >85%
- **Data Preprocessing:** >80%

---

## 🚢 Deployment

### Backend (Render)

1. **Push to GitHub** (triggers auto-deployment)
2. **Render Configuration:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.11

### Frontend (Vercel)

1. **Connect GitHub repository to Vercel**
2. **Configuration:**
   - Framework Preset: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Root Directory: `frontend`

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automatically:

- ✅ Runs tests on every push
- ✅ Checks code quality with linting
- ✅ Validates Docker builds
- ✅ Checks dependency security

---

## 🌟 Key Highlights

### For Data Science Roles
- ✅ Complete ML pipeline from raw data to production
- ✅ Advanced feature engineering with domain knowledge
- ✅ Model explainability using SHAP values
- ✅ Production-ready code with comprehensive error handling
- ✅ Performance optimization and hyperparameter tuning

### For Software Engineering Roles
- ✅ RESTful API design following best practices
- ✅ Modern React frontend with 3D animations
- ✅ Docker containerization for consistent environments
- ✅ CI/CD pipeline with automated testing
- ✅ Cloud deployment on enterprise platforms

### For Full-Stack ML Roles
- ✅ End-to-end ML system from training to deployment
- ✅ Frontend-backend integration with real-time predictions
- ✅ Scalable architecture with monitoring
- ✅ Production-grade code quality and documentation

---

## 🗺️ Roadmap

- [ ] **Authentication** - Add JWT-based user authentication
- [ ] **A/B Testing** - Implement model version comparison
- [ ] **Database** - PostgreSQL for prediction history
- [ ] **Admin Dashboard** - Model monitoring and retraining UI
- [ ] **Notifications** - Email alerts for high-risk predictions
- [ ] **Auto-Retraining** - Scheduled model updates with new data
- [ ] **Load Balancing** - Horizontal scaling for high traffic
- [ ] **API Rate Limiting** - Prevent abuse with throttling

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure:
- All tests pass (`pytest tests/`)
- Code follows PEP 8 style guide
- Documentation is updated

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Prasanth Kumar Sahu**

- 🐙 GitHub: [@PrasanthKumarS777](https://github.com/PrasanthKumarS777)
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/prasanthsahu7)
- 📧 Email: pk777sahu@gmail.com


---

## 🙏 Acknowledgments

- [XGBoost](https://xgboost.readthedocs.io/) - For the powerful gradient boosting framework
- [SHAP](https://shap.readthedocs.io/) - For model interpretability tools
- [FastAPI](https://fastapi.tiangolo.com/) - For the excellent web framework
- [React](https://react.dev/) - For the amazing UI library
- [Render](https://render.com/) & [Vercel](https://vercel.com/) - For hassle-free deployment

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and ☕ by Prasanth Kumar Sahu**

[⬆ Back to Top](#-loan-default-prediction-api-with-explainable-ai)

</div>
