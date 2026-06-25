import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

df = pd.read_csv("dataset/train.csv")

df["Dependents"] = df["Dependents"].replace("3+", "3")

numeric_cols = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop("Loan_ID", axis=1, inplace=True)

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("Accuracy:",acc)

joblib.dump(model,"model/loan_model.pkl")

# Accuracy Graph

plt.figure(figsize=(5,4))
plt.bar(["Accuracy"], [acc*100])
plt.ylabel("Percentage")
plt.title("Model Accuracy")
plt.savefig("static/accuracy.png")
plt.close()

# Confusion Matrix

cm = confusion_matrix(y_test,pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.close()

# Feature Importance

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(X.columns,importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

print("Model Saved Successfully")