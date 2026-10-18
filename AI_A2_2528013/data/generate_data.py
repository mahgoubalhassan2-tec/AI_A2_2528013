
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(student_id: str, seed: int) -> pd.DataFrame:
    """Generate reproducible synthetic crop-yield data."""

    # Verify the seed against the last four registration digits.
    digits = "".join(char for char in student_id if char.isdigit())

    if len(digits) < 4:
        raise ValueError("The student ID must contain at least four digits.")

    expected_seed = int(digits[-4:])

    if seed != expected_seed:
        raise ValueError(
            f"Seed must match the last four registration digits: "
            f"{expected_seed}."
        )

    rng = np.random.default_rng(seed)

    # Required base dataset size.
    base_rows = 240 + (seed % 41)

    df = pd.DataFrame({
        "rainfall_mm": rng.uniform(600, 1400, base_rows),
        "avg_temp_c": rng.uniform(12, 26, base_rows),
        "soil_moisture_pct": rng.uniform(20, 80, base_rows),
        "fertilizer_kg": rng.uniform(0, 250, base_rows),
        "field_size_ha": rng.uniform(0.2, 4, base_rows),
        "season": rng.choice(["A", "B"], base_rows),
        "cooperative_support": rng.integers(0, 2, base_rows),
    })

    # Season contribution: A = +0.50, B = -0.20.
    season_effect = df["season"].map({"A": 0.50, "B": -0.20})

    # Calculate the target yield using the assignment formula.
    noise = rng.normal(0, 0.80, base_rows)

    yield_tons = (
        0.004 * df["rainfall_mm"]
        + 0.06 * df["soil_moisture_pct"]
        + 0.02 * df["fertilizer_kg"]
        + 1.30 * df["field_size_ha"]
        - 0.25 * (df["avg_temp_c"] - 18).abs()
        + 0.70 * df["cooperative_support"]
        + season_effect
        + noise
    )

    df["yield_tons"] = yield_tons.clip(lower=0.50)

    # Yield per hectare determines the high-risk label.
    yield_per_hectare = df["yield_tons"] / df["field_size_ha"]
    median_yield_per_hectare = yield_per_hectare.median()

    df["high_risk"] = (
        yield_per_hectare < median_yield_per_hectare
    ).astype(int)

    # Introduce missing values at unique row positions.
    missing_count = round(0.05 * base_rows)

    soil_rows = rng.choice(
        base_rows, size=missing_count, replace=False
    )
    fertilizer_rows = rng.choice(
        base_rows, size=missing_count, replace=False
    )

    df.loc[soil_rows, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_rows, "fertilizer_kg"] = np.nan

    # Append three exact duplicate rows.
    duplicates = df.iloc[[0, 1, 2]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Musanze Harvest synthetic dataset."
    )
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    df = generate_data(args.student_id, args.seed)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / "AI_A2_2528013.csv"
    # Assignment requires numeric values to be stored to four decimal places.
    df.to_csv(output_path, index=False, float_format="%.4f")

    # Fingerprint the exact bytes written to the final CSV file.
    fingerprint = hashlib.sha256(output_path.read_bytes()).hexdigest()
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    verification_code = f"AI-A2-{args.seed:04d}"
    (artifacts_dir / "data_fingerprint.txt").write_text(
        f"Verification code: {verification_code}\n"
        f"File: data/{output_path.name}\n"
        f"SHA256: {fingerprint}\n",
        encoding="utf-8"
    )

    print(f"Verification code: AI-A2-{args.seed:04d}")
    print(f"Seed: {args.seed}")
    print(f"Base rows: {240 + args.seed % 41}")
    print(f"Rows including duplicates: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved dataset: {output_path}")


if __name__ == "__main__":
    main()
