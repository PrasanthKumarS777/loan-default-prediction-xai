import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# API URL
API_URL = "https://loan-default-prediction-xai.onrender.com"

# Title
st.title("🏦 Loan Default Prediction with XAI")
st.markdown("**Powered by XGBoost & SHAP Explainability**")

# Sidebar
st.sidebar.header("📋 Application Details")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

st.sidebar.header("💰 Financial Details")
applicant_income = st.sidebar.number_input("Applicant Income ($)", min_value=0, value=75000, step=1000)
coapplicant_income = st.sidebar.number_input("Co-applicant Income ($)", min_value=0, value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 240, 300, 480])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict button
if st.sidebar.button("🔮 Predict", use_container_width=True):
    
    # Prepare data
    data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
    
    # Make prediction
    with st.spinner("Making prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("## 📊 Prediction Result")
                    
                    if result["prediction"] == "Approved":
                        st.success(f"✅ **APPROVED**")
                        st.metric("Approval Probability", f"{result['probability']*100:.2f}%")
                    else:
                        st.error(f"❌ **REJECTED**")
                        st.metric("Rejection Probability", f"{(1-result['probability'])*100:.2f}%")
                    
                    st.info(f"⏱️ Response Time: {result.get('response_time_seconds', 0):.3f}s")
                
                with col2:
                    st.markdown("## 🔍 Top Contributing Factors")
                    
                    for factor in result["top_factors"]:
                        impact_emoji = "📈" if factor["impact"] == "Positive" else "📉"
                        impact_color = "green" if factor["impact"] == "Positive" else "red"
                        
                        st.markdown(f"{impact_emoji} **{factor['feature']}**: "
                                  f"<span style='color:{impact_color}'>{factor['contribution']:.4f}</span>",
                                  unsafe_allow_html=True)
                
                # Show detailed SHAP values
                with st.expander("📊 View Detailed SHAP Contributions"):
                    st.json(result["shap_contributions"])
                
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            st.warning("⏳ Request timed out. The free instance may be spinning up (takes ~30s). Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main area
st.markdown("---")
st.markdown("## 📖 How to Use")
st.markdown("""
1. Fill in the **Application Details** in the sidebar
2. Enter **Financial Details**
3. Click **Predict** button
4. View the prediction result and explanations
""")

st.markdown("## ℹ️ About")
st.markdown("""
This application uses **XGBoost** machine learning model to predict loan defaults.
The model is enhanced with **SHAP (SHapley Additive exPlanations)** to provide 
transparent, interpretable predictions.

**Key Features:**
- Real-time predictions
- Explainable AI with SHAP values
- Feature importance analysis
- Production-ready API backend
""")

# API Status in footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Check API Health"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=10).json()
            if health["status"] == "healthy":
                st.success("✅ API is healthy")
            else:
                st.warning("⚠️ API status unknown")
        except:
            st.error("❌ API is down")

with col2:
    if st.button("📊 View API Metrics"):
        try:
            metrics = requests.get(f"{API_URL}/metrics", timeout=10).json()
            st.json(metrics)
        except:
            st.error("❌ Unable to fetch metrics")

with col3:
    st.markdown("[📚 API Documentation](https://loan-default-prediction-xai.onrender.com/docs)")
