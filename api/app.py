from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")


model = joblib.load(r"C:\Users\user\Desktop\churn_prediction_project\models\churn.pkl")


class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: float = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "Yes"
    OnlineBackup: str = "No"
    DeviceProtection: str = "Yes"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0
    


@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input JSON to DataFrame
    df = pd.DataFrame([data.model_dump()])

    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "Churn Prediction": "Yes" if prediction == 1 else "No",
        "Churn Probability": round(float(probability), 2)
    }
