import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2] / "Database"
csv_path = base_dir / "AB_NYC_2019.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

df["name"] = df["name"].fillna("Unknown")

df["host_name"] = df["host_name"].fillna("Unknown")

df["last_review"] = df["last_review"].fillna("No Review")

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

print("\nCOLUMN NAMES BEFORE STANDARDIZATION")
print(df.columns)

df.columns = df.columns.str.lower()
print("\nCOLUMN NAMES AFTER STANDARDIZATION")
print(df.columns)

print(df.columns)

print("\nPRICE STATISTICS")
print(df["price"].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.boxplot(df["price"])
plt.title("Boxplot of Price")
plt.ylabel("Price")
plt.show()
print(df["price"].describe())

cleaned_path = base_dir / "cleaned_airbnb.csv"
cleaned_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved successfully: {cleaned_path}")
