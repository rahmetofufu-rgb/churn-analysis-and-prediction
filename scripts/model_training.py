import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# === Load original data ===
df = pd.read_csv(r"C:\Users\user\Desktop\churn_prediction_project\data\Telco-Customer-Churn.csv")

# === Basic cleaning ===
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# === Split features/target ===
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# === Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Define columns ===
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# include ALL categorical columns
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# === Preprocessing (OneHotEncoding) ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep numeric columns
)

# === Train model pipeline ===
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# === Train model ===
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}, ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}")

# === Save the entire pipeline ===
joblib.dump(model, r"C:\Users\user\Desktop\churn_prediction_project\models\churn.pkl")
print("✅ Full pipeline (preprocessor + model) saved as churn.pkl")
