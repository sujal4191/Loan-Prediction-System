from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("loan_data.csv")
print(df.head())

# ---------------- Label Encoding ----------------
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# ---------------- Features ----------------

X = df[[
    "person_income",
    "credit_score",
    "loan_amnt",
    "person_age",
    "person_gender",
    "person_emp_exp"
]]

y = df["loan_status"]

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- Feature Scaling ----------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ---------------- Model ----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# ---------------- Prediction ----------------
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Predictions:")
print(model.predict(X_test_scaled[:10]))

print("Actual:")
print(y_test.iloc[:10].values)

print(df["person_income"].describe())
print(X.columns.tolist())

# Save model & encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print(df["loan_status"].value_counts())

print(df.groupby("loan_status")[[
    "person_income",
    "credit_score",
    "loan_amnt"
]].mean())
print(model.score(X_train_scaled, y_train))
print(model.score(X_test_scaled, y_test))
print("Model Saved Successfully")
