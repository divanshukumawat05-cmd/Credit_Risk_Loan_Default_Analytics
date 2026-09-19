# Credit Risk & Loan Default Analytics

## Project Overview

Credit Risk & Loan Default Analytics is a Python project for exploring loan-default data, preparing features, training classification models, analyzing risk patterns, and providing borrower-level risk predictions through a Streamlit dashboard.

## Problem Statement

Loan default risk depends on borrower characteristics, loan details, credit history, and financial circumstances. This project organizes those data points into an analytics and machine-learning workflow that supports exploratory analysis, model evaluation, and risk prediction.

## Key Objectives

- Understand the structure and quality of the loan-default dataset.
- Explore default patterns across borrower and loan characteristics.
- Prepare features for machine-learning models.
- Compare classification models using evaluation metrics.
- Save the selected preprocessing and model pipeline for reuse.
- Provide an interactive dashboard for risk analysis and borrower prediction.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Jupyter notebooks
- SQLite

## Dataset Source

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/nikhil1e9/loan-default

The project uses the cleaned dataset stored at `outputs/cleaned_loan_default.csv`.

## Project Structure

```text
Credit_Risk_Loan_Default_Analytics/
├── data/
│   └── Loan_default.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_risk_prediction.ipynb
│   └── 07_sql_analysis.ipynb
├── outputs/
│   ├── cleaned_loan_default.csv
│   └── credit_risk_model.pkl
├── report/
├── src/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

Create and activate a virtual environment, then install the project dependencies from `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The saved model artifact at `outputs/credit_risk_model.pkl` is required by the dashboard. Run `notebooks/05_model_training.ipynb` first if the artifact is not present.

## Run the Streamlit Dashboard

From the project root, run:

```powershell
streamlit run src/app.py
```

## ML Models Used

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The models are evaluated using the logic in `notebooks/05_model_training.ipynb`. The selected preprocessing and model pipeline is stored in `outputs/credit_risk_model.pkl` for dashboard use.

## Main Dashboard Features

- Executive overview with loan count, loan amount, default rate, and average loan amount.
- Default-rate analysis by credit score group.
- Default-rate analysis by income group.
- Default-rate analysis by employment type.
- Default-rate analysis by loan purpose.
- Borrower risk prediction form.
- Model comparison summary.
