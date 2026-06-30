# Loan Status Prediction

A machine learning project that predicts whether a loan application will be **approved** or **rejected** based on applicant details such as income, education, credit history, and property area. The model is built using a **Support Vector Machine (SVM)** classifier with scikit-learn.

## 📋 Overview

Banks and financial institutions need to assess loan applications quickly and consistently. This project uses historical loan application data to train a classification model that predicts the `Loan_Status` (Approved / Rejected) for new applicants.

## 📂 Project Structure

```
Loan_Status_Prediction/
├── Loan_Status_Prediction.ipynb   # Main notebook (data processing, training, evaluation)
├── train_loan_data.csv            # Sample dataset
└── README.md
```

## 📊 Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan application ID |
| `Gender` | Male / Female |
| `Married` | Applicant's marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of the loan (in days) |
| `Credit_History` | 1 = good credit history, 0 = bad |
| `Property_Area` | Rural / Semiurban / Urban |
| `Loan_Status` | Target variable: Y (approved) / N (rejected) |


## ⚙️ Workflow

1. **Data Collection & Preprocessing**
   - Handle missing values
   - Encode categorical variables (Gender, Married, Education, Self_Employed, Property_Area, Dependents)
2. **Data Visualization**
   - Explore relationships between features (e.g. Education vs Loan Status, Marital Status vs Loan Status)
3. **Train/Test Split**
   - Stratified split to preserve class balance
4. **Model Training**
   - Support Vector Machine (linear kernel)
5. **Model Evaluation**
   - Accuracy on training and test sets
6. **Predictive System**
   - A reusable function `predict_loan_status()` that takes raw applicant details and returns a prediction

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### Run the notebook
```bash
git clone https://github.com/<your-username>/Loan_Status_Prediction.git
cd Loan_Status_Prediction
jupyter notebook Loan_Status_Prediction.ipynb
```

### Make a prediction
```python
sample_applicant = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

predict_loan_status(sample_applicant)
# Output: 'Approved (Y)'
```

## 📈 Results

On the sample dataset:

| Metric | Score |
|---|---|
| Training Accuracy | ~78.9% |
| Test Accuracy | ~76.3% |

Results will vary depending on the dataset used.

## 🛠️ Built With
- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – data processing
- [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/) – visualization
- [scikit-learn](https://scikit-learn.org/) – SVM model

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
