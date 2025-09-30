import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- Load saved model, scaler, and feature list ----------------
svm_model = joblib.load("svm_rfe_model.joblib")
scaler = joblib.load("scaler_rfe.joblib")
top_features = joblib.load("rfe_features.joblib")

# ---------------- Load new dataset ----------------
new_file = "VBM data_new.xlsx"  # Replace with your new Excel file
new_df = pd.read_excel(new_file)
new_df.columns = new_df.columns.str.strip().str.replace("'", "")

# ---------------- Prepare data ----------------
X_new = new_df[top_features]
y_true = new_df['Group']
X_new_scaled = scaler.transform(X_new)

# ---------------- Predict ----------------
y_pred = svm_model.predict(X_new_scaled)

# ---------------- Evaluation ----------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("Evaluation on New Data:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)
