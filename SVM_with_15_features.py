import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------- Load Dataset ----------------
file_name = "VBM data.xlsx"
df = pd.read_excel(file_name)
df.columns = df.columns.str.strip().str.replace("'", "")

# ---------------- Selected Features ----------------
top_features = [
    '4th Ventricle',
    'Left Lateral Ventricle',
    'Cerebellar Vermal Lobules VI-VII',
    'Right AnG angular gyrus',
    'Right Calc calcarine cortex',
    'Left Calc calcarine cortex',
    'Right Cun cuneus',
    'Left Cun cuneus',
    'Right FO frontal operculum',
    'Right LiG lingual gyrus',
    'Left MCgG middle cingulate gyrus',
    'Right MFG middle frontal gyrus',
    'Left MOrG medial orbital gyrus',
    'Right PoG postcentral gyrus',
    'Right SMG supramarginal gyrus'
]

target_col = "Group"

X = df[top_features]
y = df[target_col]

# ---------------- Scale Features ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Cross-validation ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies, precisions, recalls, f1s = [], [], [], []

print("Evaluation Metrics per Fold:")

for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    
    print(f"\nFold {fold}:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)

# ---------------- Overall Metrics ----------------
print("\n--- Overall Metrics ---")
print("Mean Accuracy:", np.mean(accuracies))
print("Mean Precision:", np.mean(precisions))
print("Mean Recall:", np.mean(recalls))
print("Mean F1-score:", np.mean(f1s))

# ---------------- Save Model & Scaler ----------------
svm_best = SVC(kernel='linear', C=1, random_state=42)
svm_best.fit(X_scaled, y)

joblib.dump(svm_best, "svm_top15_model.joblib")
joblib.dump(scaler, "scaler_top15.joblib")
joblib.dump(top_features, "top15_features.joblib")

print("\nSVM model, scaler, and top 15 features saved successfully!")
