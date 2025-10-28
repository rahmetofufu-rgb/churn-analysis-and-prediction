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
👨‍💻 Author

Firdows Rahmeto
📧 rahmetofufu@gmail.com

💻 GitHub: https://github.com/yourusername
🏁 License

This project is open for educational and portfolio use.