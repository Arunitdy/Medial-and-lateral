import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
file_path = r"VBM data.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Define features and label column
X = data.drop(columns=["Group "])   # <-- change 'Group' if your label column has another name
y = data["Group "]

# Radiologist-selected features (use exact names from Excel)

selected_features = [
 "'4th Ventricle'",
    "'Left Lateral Ventricle'",
    "'Cerebellar Vermal Lobules VI-VII'",
    "'Right AnG angular gyrus'",
    "'Right Calc calcarine cortex'",
    "'Left Calc calcarine cortex'",
    "'Right Cun cuneus'",
    "'Left Cun cuneus'",
    "'Right FO frontal operculum'",
    "'Right LiG lingual gyrus'",
    "'Left MCgG middle cingulate gyrus'",
   "'Right MFG middle frontal gyrus'",
    "'Left MOrG medial orbital gyrus'",
    "'Right PoG postcentral gyrus'",
    "'Right SMG supramarginal gyrus'"]


X_selected = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline (scaling + SVM)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# Parameter grid
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1],
    "svm__class_weight": [None, "balanced"]
}

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Results
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Test evaluation
y_pred = grid.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
