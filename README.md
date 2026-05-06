🏦 Loan Risk Analysis — ML-Based Credit Assessment
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
> **Predict loan applicant risk level (Low / Medium / High) using machine learning on 31 financial and demographic features.**
---
📌 Project Overview
This end-to-end machine learning project analyzes 2,000 loan applicants to predict their credit risk level and support loan approval decisions. It covers the complete data science pipeline — from raw data ingestion and exploratory analysis to model training, evaluation, and inference.
🎯 Problem Statement
Financial institutions need an automated, data-driven way to assess the risk of lending to an applicant. This project builds a multi-class classifier that labels each applicant as Low, Medium, or High risk based on their financial profile and personal demographics.
🏆 Key Results
Model	Accuracy	F1 Score (Weighted)	ROC-AUC
Logistic Regression	~78%	~0.77	~0.91
Random Forest	~87%	~0.87	~0.97
XGBoost ⭐	~90%	~0.90	~0.98
---
📂 Project Structure
```
loan-risk-analysis/
│
├── 📓 01_data_collection.ipynb       # Data loading, inspection & feature engineering
├── 📊 02_eda_visualization.ipynb     # EDA, charts & correlation analysis
├── 🤖 03_ml_model.ipynb              # Model training, evaluation & inference
│
├── 📁 data/
│   ├── loan_risk_data.csv            # Raw dataset (2,000 applicants, 31 features)
│   └── loan_risk_enriched.csv        # Enriched dataset (after feature engineering)
│
├── 📁 models/                        # Saved model files (generated after running 03)
│   ├── best_model.pkl
│   └── label_encoder.pkl
│
├── 📁 outputs/                       # Charts & evaluation plots (generated after running 02/03)
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files excluded from Git
└── README.md                         # You are here
```
---
🗃️ Dataset
Property	Detail
File	`loan_risk_data.csv`
Rows	2,000 applicants
Columns	31 features
Target	`risk_level` → Low (771) / Medium (729) / High (500)
Approval Split	Approved: 1,097 · Rejected: 587 · Under Review: 316
🔑 Key Features
Demographics — `age`, `gender`, `marital_status`, `education_level`, `city`, `state`
Employment — `employment_type`, `years_employed`
Financials — `annual_income_inr`, `monthly_expenses_inr`, `savings_balance_inr`, `debt_to_income_ratio`
Loan Details — `loan_amount_inr`, `loan_purpose`, `loan_tenure_months`, `interest_rate_pct`
Credit History — `credit_score`, `existing_loans`, `previous_defaults`, `num_credit_inquiries_6m`
Assets — `property_ownership`, `has_collateral`, `collateral_value_inr`
---
📓 Notebooks
`01_data_collection.ipynb` — Data Collection & Overview
Load and inspect the raw CSV dataset
Understand schema, data types, null values, and duplicates
Explore target variable and key feature distributions
Feature Engineering: Add `income_to_loan_ratio`, `savings_months_cover`, `age_group`, `application_month`
Save enriched dataset for downstream use
`02_eda_visualization.ipynb` — Exploratory Data Analysis
Risk level & approval status distributions (bar, pie, stacked charts)
Credit score analysis by risk level (box plot + KDE)
Income, loan amount & DTI ratio comparisons (interactive Plotly)
Loan purpose vs risk level heatmap
Employment type vs risk breakdown
Correlation matrix of 15 numerical features
Temporal trend of monthly applications
Geographic distribution across top 10 states
Age group & gender risk breakdown
`03_ml_model.ipynb` — Machine Learning Model
Feature selection and preprocessing pipeline (StandardScaler + OneHotEncoder)
Train/test split (80/20, stratified)
Train 3 models: Logistic Regression, Random Forest, XGBoost
Evaluate with Accuracy, F1 Score, ROC-AUC
5-Fold cross-validation
Confusion matrix visualization
Feature importance analysis (Top 20 features)
Save best model with `joblib`
Inference function to predict risk for new applicants
---
🚀 Quick Start
1. Clone the repository
```bash
git clone https://github.com/yourusername/loan-risk-analysis.git
cd loan-risk-analysis
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Launch Jupyter
```bash
jupyter notebook
```
4. Run notebooks in order
```
01_data_collection.ipynb  →  02_eda_visualization.ipynb  →  03_ml_model.ipynb
```
---
🔍 Sample Inference
After running `03_ml_model.ipynb`, you can predict risk for new applicants:
```python
import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')
le    = joblib.load('models/label_encoder.pkl')

applicant = pd.DataFrame([{
    'age': 35, 'gender': 'Male', 'employment_type': 'Salaried',
    'annual_income_inr': 1200000, 'loan_amount_inr': 4500000,
    'credit_score': 780, 'debt_to_income_ratio': 0.22,
    'previous_defaults': 0, 'existing_loans': 1,
    # ... (all required features)
}])

prediction = le.inverse_transform(model.predict(applicant))
print(f"Risk Level: {prediction[0]}")  # → 'low'
```
---
📦 Requirements
```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
imbalanced-learn
joblib
```
Install all at once:
```bash
pip install -r requirements.txt
```
---
📊 Sample Visualizations
> Charts are generated and saved to the `outputs/` folder when you run the notebooks.
Chart	Notebook
Risk Level Distribution	`02_eda_visualization`
Credit Score by Risk (Box + KDE)	`02_eda_visualization`
Loan Purpose Heatmap	`02_eda_visualization`
Correlation Matrix	`02_eda_visualization`
Confusion Matrix	`03_ml_model`
Feature Importance (Top 20)	`03_ml_model`
Model Comparison Bar Chart	`03_ml_model`
---
🛠️ Tech Stack
Tool	Purpose
Python 3.10+	Core language
Pandas & NumPy	Data manipulation
Matplotlib & Seaborn	Static visualizations
Plotly	Interactive visualizations
scikit-learn	Preprocessing & ML models
XGBoost	Gradient boosting classifier
Joblib	Model persistence
Jupyter Notebook	Development environment
---
📁 .gitignore Highlights
The following are excluded from version control:
`__pycache__/`, `*.pyc` — Python cache
`.ipynb_checkpoints/` — Jupyter auto-saves
`models/` — Trained model binaries (large files)
`outputs/` — Generated charts
`.env`, `venv/` — Environment files
---
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
Fork the repo
Create your branch: `git checkout -b feature/your-feature`
Commit your changes: `git commit -m "Add your feature"`
Push to branch: `git push origin feature/your-feature`
Open a Pull Request
---
📄 License
This project is licensed under the MIT License.
---
