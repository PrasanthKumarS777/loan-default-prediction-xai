from pydantic import BaseModel, Field
from typing import Optional, Dict


class LoanApplicationInput(BaseModel):
    """Input schema for loan application"""
    Gender: str = Field(..., description="Male or Female")
    Married: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="0, 1, 2, 3+")
    Education: str = Field(..., description="Graduate or Not Graduate")
    Self_Employed: str = Field(..., description="Yes or No")
    ApplicantIncome: float = Field(..., description="Applicant's income", gt=0)
    CoapplicantIncome: float = Field(..., description="Coapplicant's income", ge=0)
    LoanAmount: float = Field(..., description="Loan amount in thousands", gt=0)
    Loan_Amount_Term: float = Field(..., description="Loan term in months")
    Credit_History: float = Field(..., description="Credit history: 0 or 1")
    Property_Area: str = Field(..., description="Urban, Semiurban, or Rural")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Property_Area": "Urban"
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: str = Field(..., description="Approved or Rejected")
    probability: float = Field(..., description="Probability of approval")
    shap_contributions: Dict[str, float] = Field(..., description="SHAP feature contributions")
    top_factors: list = Field(..., description="Top factors influencing the decision")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
