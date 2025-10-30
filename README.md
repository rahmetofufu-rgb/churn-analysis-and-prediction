# 📊 Customer Churn Prediction Project

This project predicts telecom customer churn (whether a customer will leave the service).  
It uses **machine learning** and **deep learning** models, and provides a **FastAPI** interface for real-time predictions.

---

## 🧠 Objectives
- Clean and preprocess the Telco Customer Churn dataset.
- Perform feature engineering and create new variables.
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost).
- Evaluate and compare model performance.
- Save the best-performing model as `churn.pkl`.
- Train a deep learning model (`deep_churn_model.h5`).
- Deploy the model with a FastAPI app.
- Visualize churn trends and insights using Power BI.

---

## 🗂️ Project Structure

churn_prediction_project/
│
├─ data/
│   ├─ Telco-Customer-Churn.csv
│   ├─ summary_statistics.csv
│   └─ preprocessed_churn_data.csv
│
├─ scripts/
│   ├─ data_preprocessing.py
│   ├─ model_training.py
│   └─ deep_model.py
│
├─ notebooks/
│   ├─ 01_data_exploration.ipynb
│   ├─ 02_feature_engineering.ipynb
│   └─ 03_model_comparison.ipynb
│
├─ models/
│   ├─ churn.pkl
│   └─ deep_churn_model.h5
│
├─ api/
│   └─ app.py
│
├─ powerbi/
│   ├─ churn_dashboard.pbix      
│   └─ churn_dashboard.pdf       
│
├─ README.md
└─ requirements.txt

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
git clone https://github.com//churn_prediction_project.git

cd churn_prediction_project

### 2. Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate # On Windows
source venv/bin/activate # On Mac/Linux

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the FastAPI App
cd api
uvicorn app:app --reload

Open your browser and go to:  
👉 http://127.0.0.1:8000/docs  
to access the interactive API documentation.

---

## 🧩 Example API Request

**POST** `/predict`

Request body:
```json
{
  "tenure": 12,
  "MonthlyCharges": 65.3,
  "TotalCharges": 780.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "Yes",
  "TechSupport": "No",
  "PhoneService": "Yes",
  "Dependents": "No",
  "DeviceProtection": "No"
}
{
  "Churn Prediction": "Yes",
  "Churn Probability": 0.87
}
🧰 Tools & Libraries

Python

pandas, NumPy

scikit-learn, xgboost

TensorFlow / Keras

FastAPI, Uvicorn

Joblib

Matplotlib, Seaborn

Power BI (for data visualization)

This project focuses on analyzing customer churn data to understand why customers leave and to predict which customers are most likely to churn. The goal is to help the company improve customer retention and make informed, data-driven decisions.

---

## 📊 Project Overview

Customer churn is one of the biggest challenges for any subscription-based or service-oriented business. In this project, I explored historical customer data to:
- Identify key factors influencing churn.
- Build a predictive model to forecast potential churners.
- Provide actionable insights to help reduce churn rates.

---

## 🔍 Key Steps

1. Data Cleaning – Handled missing values, corrected inconsistencies, and prepared data for analysis.  
2. Exploratory Data Analysis (EDA) – Investigated demographic, usage, and service-related patterns.  
3. Feature Engineering – Created new variables that improved prediction accuracy.  
4. Model Building – Applied classification models (e.g., Logistic Regression, Random Forest) to predict churn.  
5. Evaluation – Assessed model accuracy, precision, recall, and F1-score.  

---

## 📈 Findings and Insights

- Contract Type & Tenure: Customers with short-term contracts were more likely to churn.  
- Monthly Charges: Higher monthly charges correlated with increased churn probability.  
- Customer Support Interaction: Customers who contacted support frequently tended to have higher churn risk.  

---

## 💡 Recommendations

Based on the findings, here are strategic actions the company could take:

1. Encourage Long-Term Contracts: Offer discounts or incentives for customers who switch from month-to-month to annual plans.  
2. Personalized Retention Campaigns: Use churn prediction results to target at-risk customers with tailored offers or support.  
3. Improve Customer Support Experience: Train support teams to identify frustration early and resolve issues quickly.  
4. Pricing Review: Consider flexible pricing or loyalty discounts for long-term customers.  
5. Customer Feedback Loop: Implement surveys to understand reasons for dissatisfaction and continuously refine services.  

---

---

## 🚀 Outcome

This project demonstrates how data analytics and machine learning can be used to:
- Anticipate customer behavior,  
- Reduce churn rates, and  
- Support strategic business decisions through actionable insights.

---


Firdows Rahmeto
📧 rahmetofufu@gmail.com

💻 GitHub: https://github.com/yourusername
🏁 License

This project is open for educational and portfolio use.
