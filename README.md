# 🏦 LoanInsight

LoanInsight is a Streamlit web app that predicts loan eligibility using a machine learning model trained on historical applicant data.

## 🔗 Live Demo

https://loaninsight-predictor.streamlit.app

## Features

- Clean, form-based UI for entering applicant details
- Input validation with clear "missing field" messages
- Instant Approved / Not Approved prediction
- Model confidence score (when supported by the underlying model)
- Expandable summary of submitted details, including computed total household income
- One-click Reset button to clear the form and start a new prediction

## How it works

The app collects 11 applicant factors used by the trained model:

| Category | Fields |
|---|---|
| Personal | Gender, Marital Status, Dependents, Education, Self Employed |
| Financial | Applicant Income, Coapplicant Income, Loan Amount, Loan Term |
| Other | Credit History, Property Area |

The applicant's name is used only to personalize the result on screen — it is not passed to the model.

Categorical fields (Gender, Married, Dependents, Education, Self Employed, Property Area) are encoded using fixed mapping dictionaries defined in `app.py`, matching scikit-learn's default alphabetical `LabelEncoder` ordering. This avoids relying on separately pickled encoder files, which can silently break if the same encoder instance was reused across columns during training.

## Project structure

```
.
├── app.py               # Streamlit application
├── loan_model.pkl        # Trained model (not included — add your own)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone the repository and navigate into the project folder.
   ```bash
   git clone https://github.com/Shrutijain-1/LoanInsight---Loan-Approval-System.git
   cd your-repo-name
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your trained model file as `loan_model.pkl` in the project root.

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Notes

- If you retrain the model with a different encoding scheme, update the `*_MAP` dictionaries near the top of `app.py` to match exactly, or predictions will be silently wrong even though the app runs without errors.
- `encoders.pkl` and `status_encoder.pkl` referenced in earlier versions of this app are no longer required.
