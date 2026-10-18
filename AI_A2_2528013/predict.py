"""Command-line and reusable inference for the Musanze Harvest service."""
from pathlib import Path
import argparse
import json
import math
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERIFICATION_CODE = "AI-A2-8013"
FEATURES = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
    "field_size_ha", "season", "cooperative_support"
]
RANGES = {
    "rainfall_mm": (600, 1400), "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80), "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4), "cooperative_support": (0, 1),
}


def validate_record(record: dict) -> None:
    """Validate all required fields and their allowed values."""
    if not isinstance(record, dict):
        raise ValueError("Input must be a JSON object.")
    missing = [key for key in FEATURES if key not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if record["season"] not in ("A", "B"):
        raise ValueError("season must be 'A' or 'B'.")
    for key, (low, high) in RANGES.items():
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number.")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")


def predict(record: dict) -> dict:
    """Return yield estimate, risk label/probability, cluster and verification code."""
    validate_record(record)
    X = pd.DataFrame([{key: record[key] for key in FEATURES}])
    regression = joblib.load(ROOT / "models" / "linear_regression.joblib")
    classifier = joblib.load(ROOT / "models" / "logistic_regression.joblib")
    clustering = joblib.load(ROOT / "models" / "clustering.joblib")

    yield_estimate = float(regression.predict(X)[0])
    risk = int(classifier.predict(X)[0])
    probabilities = classifier.predict_proba(X)[0]
    classes = list(classifier.classes_)
    high_risk_index = classes.index(1) if 1 in classes else classes.index(True)
    probability = float(probabilities[high_risk_index])

    cluster_features = [
        "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
        "fertilizer_kg", "field_size_ha", "cooperative_support"
    ]
    cluster_data = clustering["preprocessor"].transform(X[cluster_features])
    cluster = int(clustering["model"].predict(cluster_data)[0])
    return {
        "estimated_yield_tons": round(yield_estimate, 2),
        "high_risk": risk,
        "high_risk_probability": round(probability, 4),
        "cluster": cluster,
        "verification_code": VERIFICATION_CODE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict yield and farm risk from one JSON record.")
    parser.add_argument("--json", dest="json_path", help="Path to a JSON input record")
    args = parser.parse_args()
    json_path = Path(args.json_path) if args.json_path else ROOT / "samples" / "valid.json"
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        result = predict(record)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
