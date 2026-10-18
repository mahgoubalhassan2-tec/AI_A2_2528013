
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
SEED = 8013
df = pd.read_csv(ROOT / "data" / "AI_A2_2528013_clean.csv")

features = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
    "fertilizer_kg", "field_size_ha", "cooperative_support"
]

X = df[features]

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled = preprocessor.fit_transform(X)

scores = {}

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores[k] = float(silhouette_score(X_scaled, labels))
    print(f"K={k}, silhouette={scores[k]:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nBest K: {best_k}")

best_model = KMeans(
    n_clusters=best_k, random_state=SEED, n_init=10
)
df["cluster"] = best_model.fit_predict(X_scaled)

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)

import joblib

joblib.dump(
    {"preprocessor": preprocessor, "model": best_model},
    ROOT / "models" / "clustering.joblib"
)

profile = df.groupby("cluster")[features].mean().round(2)
plots_dir = ROOT / "artifacts" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

report = {
    "verification_code": f"AI-A2-{SEED:04d}",
    "selected_k": int(best_k),
    "silhouette_scores": {str(k): score for k, score in scores.items()},
    "cluster_profiles": profile.to_dict(orient="index"),
    "interpretation_note": (
        "Cluster profiles describe groups in synthetic data and are not a farm rating."
    )
}
(ROOT / "artifacts" / "clustering_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

plt.figure(figsize=(8, 5))
plt.bar([str(k) for k in scores], list(scores.values()))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Clustering model selection")
plt.tight_layout()
plt.savefig(plots_dir / "clusters.png", dpi=160)
plt.savefig(ROOT / "artifacts" / "clusters.png", dpi=160)
plt.close()

print("\nCluster profiles:")
print(profile)
print("\nClustering report and plot saved.")
