import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# Load preprocessed data
df = pd.read_csv(r"C:\Users\user\Desktop\churn_prediction_project\data\preprocessed_churn_data.csv")
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models and parameter grids
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {'penalty':['l1','l2'],'C':[0.01,0.1,1],'solver':['liblinear','saga']}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {'n_estimators':[100,200],'max_depth':[None,5,10],'min_samples_split':[2,5]}
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "params": {'n_estimators':[100,200],'max_depth':[3,5,7],'learning_rate':[0.01,0.1,0.2]}
    }
}

results = {}
best_models = {}
plt.figure(figsize=(8,6))

# Run GridSearch, evaluate, plot ROC
for name, mp in models.items():
    print(f"\n🔹 GridSearch for {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Print results
for name, metric in results.items():
    print(f"\n{name}:")
    for m, v in metric.items():
        print(f"  {m}: {v:.3f}")

# Save best model
best_model_name = max(results, key=lambda x: results[x]['ROC-AUC'])
joblib.dump(best_models[best_model_name], "C:/Users/user/Desktop/churn_prediction_project/models/churn.pkl")
print(f"🏆 Best model: {best_model_name} saved as models/churn.pkl")
