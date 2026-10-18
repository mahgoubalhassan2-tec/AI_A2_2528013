"""Validate the generated dataset and guard against target leakage."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 8013
CODE = f"AI-A2-{SEED:04d}"
REQUIRED_COLUMNS = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
    "field_size_ha", "season", "cooperative_support", "yield_tons", "high_risk"
]
RANGES = {
    "rainfall_mm": (600, 1400), "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80), "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0), "cooperative_support": (0, 1),
    "yield_tons": (0.5, None), "high_risk": (0, 1),
}
FEATURES = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
    "field_size_ha", "season", "cooperative_support"
]


def validate_dataset(path: Path) -> dict:
    """Validate schema, ranges, missingness, duplicates, and feature leakage."""
    df = pd.read_csv(path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.columns.tolist() != REQUIRED_COLUMNS:
        raise ValueError("Dataset columns or column order do not match the required schema.")

    range_errors = {}
    for column, (low, high) in RANGES.items():
        series = df[column].dropna()
        invalid = series < low
        if high is not None:
            invalid = invalid | (series > high)
        if invalid.any():
            range_errors[column] = int(invalid.sum())
    invalid_season = int((~df["season"].dropna().isin(["A", "B"])).sum())
    if invalid_season:
        range_errors["season"] = invalid_season
    if range_errors:
        raise ValueError(f"Out-of-range values detected: {range_errors}")

    leaked_features = sorted(set(FEATURES) & {"yield_tons", "high_risk", "yield_per_hectare"})
    if leaked_features:
        raise ValueError(f"Target leakage in feature list: {leaked_features}")

    report = {
        "verification_code": CODE,
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "range_checks": "passed for all observed non-missing values",
        "target_leakage_check": "passed; target columns are not in feature list",
        "missing_values_policy": "preserved for train-only pipeline imputation",
    }
    out = ROOT / "artifacts" / "data_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    validate_dataset(ROOT / "data" / "AI_A2_2528013.csv")
