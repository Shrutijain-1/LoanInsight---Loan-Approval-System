import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# 1. Load dataset
df = pd.read_csv("loan_data.csv")

# 2. Remove loan_id
df = df.drop("loan_id", axis=1)

# 3. Columns that should contain numbers
numeric_columns = [
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "loan_amount_term",
    "credit_history"
]

# Convert these columns to numbers
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 4. Fill missing values
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing text values
categorical_columns = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "property_area"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

## 5. Convert text columns into numbers
encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# Convert loan_status into numbers
status_encoder = LabelEncoder()
df["loan_status"] = status_encoder.fit_transform(df["loan_status"])

# 7. Separate input and output
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 9. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 10. Train model
model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# 13. Save model
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(status_encoder, "status_encoder.pkl")

print("Model saved successfully!")
print("Encoders saved successfully!")