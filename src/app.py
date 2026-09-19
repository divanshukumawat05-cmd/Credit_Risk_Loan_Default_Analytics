from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'outputs' / 'cleaned_loan_default.csv'
MODEL_PATH = ROOT / 'outputs' / 'credit_risk_model.pkl'

numeric_features = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'MonthsEmployed',
    'NumCreditLines',
    'InterestRate',
    'LoanTerm',
    'DTIRatio',
]
binary_features = ['HasMortgage', 'HasDependents', 'HasCoSigner']
categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
selected_features = numeric_features + binary_features + categorical_features


@st.cache_data
def load_clean_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            'Saved model not found. Run notebooks/05_model_training.ipynb first '
            'to create outputs/credit_risk_model.pkl.'
        )

    with MODEL_PATH.open('rb') as model_file:
        return pickle.load(model_file)


@st.cache_resource
def load_model_and_data():
    df = load_clean_dataset()
    model_pipeline = load_model_pipeline()
    comparison_df = model_pipeline.comparison_df
    selected_model_name = model_pipeline.selected_model_name
    return df, comparison_df, model_pipeline, selected_model_name


def calculate_default_rate(df, by_col):
    grouped = (
        df.groupby(by_col, dropna=False)
        .agg(
            total_loans=('Default', 'size'),
            default_count=('Default', 'sum'),
        )
        .reset_index()
    )
    grouped['default_rate'] = grouped['default_count'] / grouped['total_loans']
    return grouped


def default_rate_by_credit_score(df):
    df_copy = df.copy()
    bins = [0, 549, 649, 749, float('inf')]
    labels = ['Below 550', '550-649', '650-749', '750+']
    df_copy['credit_score_group'] = pd.cut(
        df_copy['CreditScore'],
        bins=bins,
        labels=labels,
        right=False,
    )
    return calculate_default_rate(df_copy, 'credit_score_group')


def default_rate_by_income(df):
    df_copy = df.copy()
    bins = [0, 25000, 50000, 75000, 100000, 125000, float('inf')]
    labels = ['0-24k', '25k-49k', '50k-74k', '75k-99k', '100k-124k', '125k+']
    df_copy['income_group'] = pd.cut(
        df_copy['Income'],
        bins=bins,
        labels=labels,
        right=False,
    )
    return calculate_default_rate(df_copy, 'income_group')


def classify_risk(probability):
    if probability < 0.10:
        return 'Low'
    if probability < 0.30:
        return 'Medium'
    return 'High'


def predict_single_borrower(model_pipeline, borrower_input):
    record = pd.DataFrame([borrower_input])
    feature_order = selected_features
    record = record[feature_order].copy()

    for col in binary_features:
        if col in record.columns:
            record[col] = record[col].map(
                {
                    'Yes': 'Yes',
                    'No': 'No',
                    1: 'Yes',
                    0: 'No',
                    True: 'Yes',
                    False: 'No',
                }
            )

    probability = model_pipeline.predict_proba(record)[0, 1]
    prediction = int(model_pipeline.predict(record)[0])

    return {
        'Predicted Default': prediction,
        'Default Probability': float(probability),
        'Risk Level': classify_risk(float(probability)),
    }


st.set_page_config(page_title='Credit Risk & Loan Default Analytics', layout='centered')
st.markdown(
    """
    <style>
    [data-testid='stAppViewContainer'] .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid='stMetric'] {
        min-height: 105px;
        padding: 1rem 1.1rem;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        background: #f8fafc;
    }
    [data-testid='stMetricLabel'] {
        font-size: 0.9rem;
        font-weight: 600;
    }
    [data-testid='stMetricValue'] {
        font-size: 1.65rem;
    }
    h1 {
        margin-bottom: 1.5rem;
    }
    h2 {
        margin-top: 2rem;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #d9e2ec;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('Credit Risk & Loan Default Analytics')

with st.spinner('Loading dataset and trained model...'):
    df, comparison_df, model_pipeline, selected_model_name = load_model_and_data()

# Executive Overview
st.subheader('1. Executive Overview')

summary = {
    'Total Loans': len(df),
    'Total Loan Amount': df['LoanAmount'].sum(),
    'Default Rate': df['Default'].mean(),
    'Average Loan Amount': df['LoanAmount'].mean(),
}

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Loans', f"{summary['Total Loans']:,}")
col2.metric('Total Loan Amount', f"${summary['Total Loan Amount']:,.0f}")
col3.metric('Default Rate', f"{summary['Default Rate'] * 100:.2f}%")
col4.metric('Average Loan Amount', f"${summary['Average Loan Amount']:,.0f}")

# Risk Analysis
st.subheader('2. Risk Analysis')

credit_group = default_rate_by_credit_score(df)
fig_credit = px.bar(
    credit_group,
    x='credit_score_group',
    y='default_rate',
    title='Default Rate by Credit Score Group',
    labels={'credit_score_group': 'Credit Score Group', 'default_rate': 'Default Rate'},
    color='default_rate',
    color_continuous_scale='RdYlGn_r',
)
fig_credit.update_layout(template='plotly_white')
fig_credit.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=45))

income_group = default_rate_by_income(df)
fig_income = px.bar(
    income_group,
    x='income_group',
    y='default_rate',
    title='Default Rate by Income Group',
    labels={'income_group': 'Income Group', 'default_rate': 'Default Rate'},
    color='default_rate',
    color_continuous_scale='RdYlGn_r',
)
fig_income.update_layout(template='plotly_white')
fig_income.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=45))

employment_rate = calculate_default_rate(df, 'EmploymentType')
fig_employment = px.bar(
    employment_rate,
    x='EmploymentType',
    y='default_rate',
    title='Default Rate by Employment Type',
    labels={'EmploymentType': 'Employment Type', 'default_rate': 'Default Rate'},
    color='default_rate',
    color_continuous_scale='RdYlGn_r',
)
fig_employment.update_layout(template='plotly_white')
fig_employment.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=45))

purpose_rate = calculate_default_rate(df, 'LoanPurpose')
fig_purpose = px.bar(
    purpose_rate,
    x='LoanPurpose',
    y='default_rate',
    title='Default Rate by Loan Purpose',
    labels={'LoanPurpose': 'Loan Purpose', 'default_rate': 'Default Rate'},
    color='default_rate',
    color_continuous_scale='RdYlGn_r',
)
fig_purpose.update_layout(template='plotly_white')
fig_purpose.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=45))

col_a, col_b = st.columns(2)
col_a.plotly_chart(fig_credit, use_container_width=True)
col_b.plotly_chart(fig_income, use_container_width=True)
col_c, col_d = st.columns(2)
col_c.plotly_chart(fig_employment, use_container_width=True)
col_d.plotly_chart(fig_purpose, use_container_width=True)

# Borrower Risk Prediction
st.subheader('3. Borrower Risk Prediction')

with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=35)
        income = st.number_input('Income', min_value=0, value=75000)
        loan_amount = st.number_input('Loan Amount', min_value=0, value=40000)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=680)
        months_employed = st.number_input('Months Employed', min_value=0, value=18)
        num_credit_lines = st.number_input('Number of Credit Lines', min_value=0, value=2)
        interest_rate = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, value=9.0)
        loan_term = st.number_input('Loan Term (Months)', min_value=1, value=36)
        dti_ratio = st.number_input('DTI Ratio', min_value=0.0, max_value=1.0, value=0.30)

    with col2:
        education = st.selectbox('Education', ['High School', 'Bachelor\'s', "Master's", 'Doctorate'])
        employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Separated'])
        has_mortgage = st.selectbox('Has Mortgage', ['Yes', 'No'])
        has_dependents = st.selectbox('Has Dependents', ['Yes', 'No'])
        loan_purpose = st.selectbox('Loan Purpose', ['Auto', 'Business', 'Education', 'Home', 'Other'])
        has_cosigner = st.selectbox('Has Co-Signer', ['Yes', 'No'])

    submitted = st.form_submit_button('Predict Risk')

if submitted:
    borrower_input = {
        'Age': int(age),
        'Income': int(income),
        'LoanAmount': int(loan_amount),
        'CreditScore': int(credit_score),
        'MonthsEmployed': int(months_employed),
        'NumCreditLines': int(num_credit_lines),
        'InterestRate': float(interest_rate),
        'LoanTerm': int(loan_term),
        'DTIRatio': float(dti_ratio),
        'Education': education,
        'EmploymentType': employment_type,
        'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage,
        'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner,
    }

    prediction_result = predict_single_borrower(model_pipeline, borrower_input)
    probability = prediction_result['Default Probability']

    st.metric('Probability of Default', f'{probability * 100:.2f}%')
    st.metric('Predicted Default', prediction_result['Predicted Default'])
    st.metric('Risk Level', prediction_result['Risk Level'])

    # Show model ranking in sidebar or below
    st.caption(f'Using selected model: {selected_model_name}')

    st.dataframe(pd.DataFrame([prediction_result]))

st.sidebar.header('Model Summary')
st.sidebar.dataframe(comparison_df.sort_values(by=['F1-score', 'ROC-AUC', 'Recall', 'Precision'], ascending=False), use_container_width=True)
