import pandas as pd

df = pd.read_csv(r"c:\Users\krush\Desktop\OIBSIP\project3\Database\datasets\apps.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATASET INFO")
print(df.info())
print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())



df = df.drop("Unnamed: 0", axis=1)

print("\nCOLUMN NAMES AFTER CLEANING")
print(df.columns.tolist())

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

print("\nMISSING VALUES AFTER RATING CLEANING")
print(df["Rating"].isnull().sum())


print(df.duplicated().sum())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

import matplotlib.pyplot as plt

category_count = df["Category"].value_counts()

print("\nTOP APP CATEGORIES")
print(category_count.head(10))

category_count.head(10).plot(kind="bar")

plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")

plt.show()

print("\nRATING STATISTICS")
print(df["Rating"].describe())

import matplotlib.pyplot as plt

df["Rating"].hist(bins=20)

plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Apps")

plt.show()

df["Installs"] = df["Installs"].str.replace(",", "", regex=False)
df["Installs"] = df["Installs"].str.replace("+", "", regex=False)

df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

print("\nINSTALLS STATISTICS")
print(df["Installs"].describe())

top_apps = df[["App", "Installs"]].sort_values(
    by="Installs",
    ascending=False
)

print("\nTOP 10 MOST INSTALLED APPS")
print(top_apps.head(10))


df["Installs"].hist(bins=20)

plt.title("Distribution of App Installs")
plt.xlabel("Installs")
plt.ylabel("Number of Apps")

plt.show()

print("\nFREE VS PAID APPS")
print(df["Type"].value_counts())

df["Type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Free vs Paid Apps")
plt.ylabel("")
plt.show()


print("\nPRICE STATISTICS")
print(df["Price"].value_counts().head(10))

df["Price"] = df["Price"].str.replace("$", "", regex=False)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

print("\nPRICE COLUMN CLEANED")
print(df["Price"].describe())


top_expensive = df[["App", "Price"]].sort_values(
    by="Price",
    ascending=False
)

print("\nTOP 10 MOST EXPENSIVE APPS")
print(top_expensive.head(10))



df["Price"].hist(bins=20)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Apps")

plt.show()


reviews_df = pd.read_csv(r"c:\Users\krush\Desktop\OIBSIP\project3\Database\datasets\user_reviews.csv")

print("\nUSER REVIEWS DATASET")
print(reviews_df.head())

print("\nUSER REVIEWS SHAPE")
print(reviews_df.shape)



print("\nSENTIMENT DISTRIBUTION")
print(reviews_df["Sentiment"].value_counts())



reviews_df["Sentiment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("User Sentiment Distribution")
plt.ylabel("")
plt.show()



print("\nSENTIMENT DISTRIBUTION")
print(reviews_df["Sentiment"].value_counts())


print("\nTOP 10 CATEGORIES")
print(df["Category"].value_counts().head(10))

print("\nFREE VS PAID")
print(df["Type"].value_counts())