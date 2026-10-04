# Lending Club Loan Default Prediction

Deep learning model to predict loan default risk using Lending Club's historical loan data (2007–2015). Built with Keras/TensorFlow, this project tackles a highly imbalanced, finance-domain binary classification problem.

## Objective

Predict whether a loan will default (`not.fully.paid`) using borrower credit and loan attributes, so Lending Club can better assess default risk on future loans.

## Dataset

~9,500+ historical loans with features including:

| Feature | Description |
|---|---|
| `credit.policy` | 1 if borrower meets LendingClub's underwriting criteria |
| `purpose` | Loan purpose (credit card, debt consolidation, small business, etc.) |
| `int.rate` | Interest rate assigned (higher = riskier borrower) |
| `installment` | Monthly installment owed |
| `log.annual.inc` | Log of self-reported annual income |
| `dti` | Debt-to-income ratio |
| `fico` | Borrower's FICO credit score |
| `days.with.cr.line` | Days the borrower has had a credit line |
| `revol.bal` / `revol.util` | Revolving balance / credit utilization rate |
| `inq.last.6mths` | Credit inquiries in the last 6 months |
| `delinq.2yrs` | Times 30+ days past due in the last 2 years |
| `pub.rec` | Derogatory public records (bankruptcies, liens) |
| `not.fully.paid` | **Target** — 1 if loan defaulted/charged off, 0 if fully paid |

**Key challenge:** the dataset is highly imbalanced — the large majority of loans are fully paid, so naively optimizing for accuracy causes models to miss defaults almost entirely.

## Approach

1. **Feature Transformation** — Label-encoded the categorical `purpose` field into numeric form
2. **Exploratory Data Analysis** — Correlation heatmap across all numeric features, distribution plots, and default rate breakdowns by `purpose` and `credit.policy`
3. **Class Imbalance Handling** — Applied **SMOTE** (Synthetic Minority Over-sampling) to balance the `not.fully.paid` classes before training
4. **Modeling** — Built a Keras Sequential neural network:
   - Dense input layer (ReLU) sized to the feature count
   - Dense hidden layer (50 units, ReLU)
   - Dense output layer (1 unit, sigmoid) for binary classification
   - Compiled with Adam optimizer and binary cross-entropy loss
5. **Evaluation** — Assessed on a held-out, stratified test split using accuracy, with classification report and confusion matrix

## Beyond the Baseline: Risk-Optimized Architecture

[`reports/Lending_Club_Model_Analysis_Report.docx`](reports/Lending_Club_Model_Analysis_Report.docx) documents a deeper architectural analysis for production-grade risk modeling on this same problem, addressing why standard MLPs plateau around 80–85% accuracy on imbalanced credit data. It outlines six refinements beyond the baseline notebook:

1. Tracking **Recall, Precision, and AUC** instead of raw accuracy, since accuracy alone can hide a 0% recall on defaults
2. **Class weighting** to penalize misclassified defaults more heavily during training
3. **Domain-driven engineered ratios** — installment-to-income utilization, a financial stress index (DTI × revolving utilization), and interest-per-FICO risk premium
4. **Batch Normalization** after each dense layer to stabilize training and prevent internal covariate shift
5. **Early stopping** on validation loss to prevent overfitting from excessive epochs
6. **Decision threshold calibration** (lowering from 0.5 to ~0.35–0.40) to catch more high-risk loans in underwriting, trading some accuracy for better default recall

## Tech Stack

- **Python 3**, Jupyter Notebook
- `pandas`, `numpy` for data manipulation
- `scikit-learn` — `LabelEncoder`, `train_test_split`, evaluation metrics
- `imbalanced-learn` (SMOTE) for class balancing
- `tensorflow` / `keras` for the deep learning model
- `matplotlib`, `seaborn` for visualization

## Project Structure

```
├── notebooks/
│   └── Lending_Club_Loan_Data_Analysis.ipynb   # EDA, SMOTE balancing & Keras model
├── reports/
│   └── Lending_Club_Model_Analysis_Report.docx  # Technical deep-dive on risk-optimized architecture
├── data/
│   └── loan_data.csv                             # Historical Lending Club loan data (2007–2015)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn
jupyter notebook notebooks/Lending_Club_Loan_Data_Analysis.ipynb
```

## Author

Sanjay Gosai
*Deep Learning with Keras and TensorFlow — Course-End Project*
