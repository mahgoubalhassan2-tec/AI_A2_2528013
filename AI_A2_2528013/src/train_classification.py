
from pathlib import Path

import joblib
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "AI_A2_2528013_clean.csv")

features = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "season",
    "cooperative_support"
]
X = df[features]
y = df["high_risk"]

numeric = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "cooperative_support"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), ["season"])
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8013, stratify=y
)

models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=8013
    )
}

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    if len(set(y_test)) == 2:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )

    results[name] = metrics
    print(name, metrics)

    if name != "baseline":
        joblib.dump(pipeline, ROOT / "models" / f"{name}.joblib")

report = {
    "verification_code": "AI-A2-8013",
    "target": "high_risk",
    "positive_class": 1,
    "test_size": 0.2,
    "random_state": 8013,
    "models": results,
    "selected_model": "logistic_regression",
    "selection_reason": (
        "Logistic regression matches random forest on accuracy, precision, recall, and F1 "
        "in this split, and has the higher ROC-AUC."
    ),
    "confusion_matrix_labels": {"0": "not high risk", "1": "high risk"},
}
(ROOT / "artifacts" / "classification_metrics.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
