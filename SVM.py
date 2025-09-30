import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_name = "VBM data.xlsx"
df = pd.read_excel(file_name)

# Target column (update if your label column has a different name, e.g., "Diagnosis" or "Group")
target_col = "Group"

# Features to use
selected_features = [
    "Cerebellar Vermal Lobules VI-VII",
    "Left PCu precuneus",
    "Right LiG lingual gyrus",
    "Right PoG postcentral gyrus"
]

# Subset dataset
X = df[selected_features]
y = df[target_col]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build SVM model
svm = SVC(kernel="linear", C=1, random_state=42)
svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
