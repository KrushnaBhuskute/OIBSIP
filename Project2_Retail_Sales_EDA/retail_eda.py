import sys
from pathlib import Path
import pandas as pd

def find_csv_file():
	# Path relative to this script file
	base = Path(__file__).resolve().parent
	candidates = [
		base.parent / "Database" / "retail_sales_dataset.csv",
		base / "../Database/retail_sales_dataset.csv",
		Path.cwd() / "project2" / "Database" / "retail_sales_dataset.csv",
		Path.cwd() / "Database" / "retail_sales_dataset.csv",
	]
	for p in candidates:
		p = p.resolve()
		if p.exists():
			return p
	return None


csv_path = find_csv_file()
if csv_path is None:
	print("Error: could not find 'retail_sales_dataset.csv'. Tried common locations.")
	sys.exit(1)

try:
	df = pd.read_csv(csv_path)
except Exception as e:
	print("Error reading CSV:", e)
	sys.exit(1)

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(list(df.columns))

print("\nDATASET INFO")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

if __name__ == '__main__':
	pass

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())
df.duplicated().sum()

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())


df["Date"] = pd.to_datetime(df["Date"])

print("\nDATE CONVERTED SUCCESSFULLY")
print(df["Date"].head())

import matplotlib.pyplot as plt

monthly_sales = df.groupby(df["Date"].dt.month)["Total Amount"].sum()

plt.figure(figsize=(8,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

# Ensure target directory exists (create if missing)
save_path = (Path(__file__).resolve().parent / ".." / "Screenshots" / "monthly_sales_trend.png").resolve()
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path)
plt.show()
product_sales = df.groupby("Product Category")["Total Amount"].sum()

print("\nPRODUCT CATEGORY SALES")
print(product_sales)

product_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Sales")

plt.show()

gender_sales = df.groupby("Gender")["Total Amount"].sum()

print("\nGENDER SALES")
print(gender_sales)

gender_sales.plot(kind="bar")

plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Sales")

plt.show()


try:
	import seaborn as sns

	plt.figure(figsize=(8,5))

	sns.heatmap(
		df[["Age","Quantity","Price per Unit","Total Amount"]].corr(),
		annot=True
	)

	plt.title("Correlation Heatmap")
	# Save heatmap to Screenshots directory to avoid GUI blocking
	heatmap_path = (Path(__file__).resolve().parent / ".." / "Screenshots" / "correlation_heatmap.png").resolve()
	heatmap_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(heatmap_path)
	plt.show()
except ImportError:
	print("Seaborn not installed; skipping correlation heatmap. Install with: pip install seaborn")
	

print(product_sales)