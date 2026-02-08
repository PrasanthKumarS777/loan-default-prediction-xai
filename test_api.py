import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health Check
def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("✅ Health Check:", response.json())
    return response.status_code == 200

# Test 2: Single Prediction
def test_single_prediction():
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
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("✅ Single Prediction:", response.json())
    return response.status_code == 200

# Test 3: Batch Prediction
def test_batch_prediction():
    data = {
        "applications": [
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
            },
            {
                "Gender": "Female",
                "Married": "No",
                "Dependents": "1",
                "Education": "Graduate",
                "Self_Employed": "Yes",
                "ApplicantIncome": 45000,
                "CoapplicantIncome": 15000,
                "LoanAmount": 100,
                "Loan_Amount_Term": 180,
                "Credit_History": 0.0,
                "Property_Area": "Rural"
            },
            {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "2",
                "Education": "Not Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 30000,
                "CoapplicantIncome": 20000,
                "LoanAmount": 80,
                "Loan_Amount_Term": 240,
                "Credit_History": 1.0,
                "Property_Area": "Semiurban"
            }
        ]
    }
    response = requests.post(f"{BASE_URL}/batch_predict", json=data)
    print("✅ Batch Prediction:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

if __name__ == "__main__":
    print("\n🧪 Running API Tests...\n")
    
    results = {
        "Health Check": test_health(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction()
    }
    
    print("\n📊 Test Results:")
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
