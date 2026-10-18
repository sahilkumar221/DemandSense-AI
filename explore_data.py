
import pandas as pd

# Load the bike rental dataset
df = pd.read_csv("data/hour.csv")

# 1. Dataset dimensions
print("Dataset dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\nColumn names:")
print(df.columns.tolist())

# 3. Data types and non-null counts
print("\nDataset information:")
df.info()

# 4. Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 5. Basic statistics
print("\nBasic statistical summary:")
print(df.describe())
