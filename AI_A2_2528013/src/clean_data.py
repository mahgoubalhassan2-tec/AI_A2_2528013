
from pathlib import Path

import pandas as pd


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Remove exact duplicate rows and save the cleaned dataset.
    Missing values are preserved for later preprocessing.
    """

    # Load the generated CSV file.
    df = pd.read_csv(input_path)

    # Record the original number of rows.
    original_rows = len(df)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Create the output directory if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the cleaned dataset.
    df.to_csv(output_path, index=False)

    # Display a summary.
    print(f"Original rows: {original_rows}")
    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Duplicates removed: {original_rows - len(df)}")
    print("\nMissing values remaining:")
    print(df.isna().sum())
    print(f"\nCleaned dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_csv = project_root / "data" / "AI_A2_2528013.csv"
    output_csv = project_root / "data" / "AI_A2_2528013_clean.csv"

    clean_data(input_csv, output_csv)
