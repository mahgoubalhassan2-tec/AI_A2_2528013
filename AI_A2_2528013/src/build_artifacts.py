"""Build required evaluation artifacts after model training."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
SEED = 8013
CODE = f"AI-A2-{SEED:04d}"
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "AI_A2_2528013_clean.csv")
features = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "season", "cooperative_support"
]
X = df[features]
y_reg = df["yield_tons"]
y_cls = df["high_risk"].astype(int)

# Use the same stratified row split for both supervised tasks.
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=SEED, stratify=y_cls
)
_, X_test_cls, _, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls
)

reg = joblib.load(ROOT / "models" / "linear_regression.joblib")
clf = joblib.load(ROOT / "models" / "logistic_regression.joblib")
reg_pred = reg.predict(X_test)
cls_pred = clf.predict(X_test_cls)
cls_prob = clf.predict_proba(X_test_cls)[:, 1]

# Cluster assignment is generated with the saved fitted preprocessing and model.
cluster_bundle = joblib.load(ROOT / "models" / "clustering.joblib")
cluster_features = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "cooperative_support"
]
all_cluster_features = df[cluster_features]
cluster_labels = cluster_bundle["model"].predict(
    cluster_bundle["preprocessor"].transform(all_cluster_features)
)

predictions = pd.DataFrame({
    "row_id": X_test.index.astype(int),
    "actual_yield_tons": y_test_reg.to_numpy(),
    "predicted_yield_tons": reg_pred,
    "absolute_error_tons": abs(y_test_reg.to_numpy() - reg_pred),
    "actual_high_risk": y_test_cls.to_numpy(),
    "predicted_high_risk": cls_pred,
    "high_risk_probability": cls_prob,
    "cluster": cluster_labels[X_test.index.to_numpy()],
    "verification_code": CODE,
})
predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False, float_format="%.4f")

# Regression residual plot.
residuals = y_test_reg.to_numpy() - reg_pred
plt.figure(figsize=(7, 5))
plt.scatter(reg_pred, residuals, alpha=0.8)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted yield (tons)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Regression residuals: held-out test set")
plt.tight_layout()
plt.savefig(PLOTS / "residuals.png", dpi=160)
plt.close()

# Classification confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test_cls, cls_pred, display_labels=["Low risk", "High risk"], ax=ax
)
ax.set_title("Logistic regression confusion matrix")
fig.tight_layout()
fig.savefig(PLOTS / "confusion_matrix.png", dpi=160)
plt.close(fig)

# Keep compatibility copies at the artifact root as well.
import shutil
shutil.copy2(PLOTS / "residuals.png", ARTIFACTS / "residuals.png")
shutil.copy2(PLOTS / "confusion_matrix.png", ARTIFACTS / "confusion_matrix.png")
shutil.copy2(PLOTS / "clusters.png", ARTIFACTS / "clusters.png")

summary = {
    "verification_code": CODE,
    "test_rows": int(len(predictions)),
    "regression_model": "linear_regression",
    "classification_model": "logistic_regression",
    "prediction_file": "artifacts/test_predictions.csv",
    "plots": [
        "artifacts/plots/residuals.png",
        "artifacts/plots/confusion_matrix.png",
        "artifacts/plots/clusters.png",
    ],
}
(ARTIFACTS / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Built evaluation artifacts for {len(predictions)} test rows.")
print("Created test_predictions.csv, evaluation summary, and required plots.")
