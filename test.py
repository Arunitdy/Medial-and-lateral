import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------- Load Dataset ----------------
file_name = "VBM data.xlsx"
df = pd.read_excel(file_name)
df.columns = df.columns.str.strip().str.replace("'", "")

# ---------------- Features & Target ----------------
target_col = "Group"
non_features = ['Group', 'Age', 'Gender']
feature_columns = [col for col in df.columns if col not in non_features]

X = df[feature_columns]
y = df[target_col]

# ---------------- Scale Features ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- RFE with SVM ----------------
svm_estimator = SVC(kernel='linear', C=1, random_state=42)
rfe = RFE(estimator=svm_estimator, n_features_to_select=15, step=1)
rfe.fit(X_scaled, y)

# Top 15 Features
top_features = [f for f, s in zip(feature_columns, rfe.support_) if s]
print("Top 15 Features selected by RFE with SVM:")
for f in top_features:
    print("-", f)

# Train SVM on selected features
X_rfe_scaled = scaler.fit_transform(X[top_features])
svm_best = SVC(kernel='linear', C=1, random_state=42)
svm_best.fit(X_rfe_scaled, y)

# ---------------- Cross-validation ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_best, X_rfe_scaled, y, cv=cv)
print("\nSVM Accuracy (5-fold CV):", cv_scores.mean())
print("Accuracy per fold:", cv_scores)

# ---------------- Save Model, Scaler, Features ----------------
joblib.dump(svm_best, "svm_rfe_model.joblib")
joblib.dump(scaler, "scaler_rfe.joblib")
joblib.dump(top_features, "rfe_features.joblib")

print("\nSVM model, scaler, and feature list saved successfully!")
