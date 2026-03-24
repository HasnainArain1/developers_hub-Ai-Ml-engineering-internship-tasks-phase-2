# Telco Customer Churn Prediction Pipeline

## Objective
Predict which telecom customers are likely to churn using
a production-ready Scikit-learn Pipeline.

## Methodology
- Dataset: IBM Telco Churn (7043 customers, 20 features)
- Preprocessing: SimpleImputer, StandardScaler, OneHotEncoder
- Models: Logistic Regression and Random Forest
- Tuning: GridSearchCV with 5-fold StratifiedKFold
- Export: Best pipeline saved using joblib

## Key Results
- Best Model: Logistic Regression
- AUC-ROC: 0.8462
- Top churn predictors: tenure, TotalCharges, Contract type, MonthlyCharges