
import pandas as pd
from pathlib import Path

file_path = Path("data") / "hour.csv"

if file_path.exists():
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())
else:
    print("Dataset not found!")
    print("Check that hour.csv is inside the data folder.")
