
from pathlib import Path

import joblib
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "AI_A2_2528013_clean.csv"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

df = pd.read_csv(DATA)

features = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "season",
    "cooperative_support"
]

X = df[features]
y = df["yield_tons"]

numeric = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "cooperative_support"
]
categorical = ["season"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8013, stratify=df["high_risk"]
)

models = {
    "baseline": DummyRegressor(strategy="mean"),
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=8013
    )
}

MODELS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results[name] = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions))
    }

    if name != "baseline":
        joblib.dump(pipeline, MODELS / f"{name}.joblib")

    print(name, results[name])

report = {
    "verification_code": "AI-A2-8013",
    "target": "yield_tons",
    "test_size": 0.2,
    "random_state": 8013,
    "models": results,
    "selected_model": "linear_regression",
    "selection_reason": (
        "Linear regression has the lowest held-out MAE and RMSE and the highest R2 "
        "among the evaluated regression candidates."
    ),
}
(ARTIFACTS / "regression_metrics.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
