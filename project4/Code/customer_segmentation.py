import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / ".." / "Database" / "OnlineRetail.csv"

    df = pd.read_csv(data_path, encoding="latin1")

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

    print("\nDESCRIPTIVE STATISTICS")
    print(df.describe())

    print("\n==========================================")
    print("       DATASET CLEANING")
    print("==========================================")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    print("Rows after removing missing CustomerID and cancelled orders:")
    print(df.shape)

    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    print("\nSNAPSHOT DATE")
    print(snapshot_date)

    rfm = df.groupby("CustomerID").agg(
        InvoiceDate=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        InvoiceNo=("InvoiceNo", "nunique"),
        TotalAmount=("TotalAmount", "sum"),
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    print("\nRFM TABLE")
    print(rfm.head())

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    print("\nSTANDARDIZED RFM DATA")
    print(rfm_scaled[:5])

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    print("\nWCSS FOR ELBOW METHOD")
    print(wcss)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.savefig("elbow_plot.png")
    print("\nElbow plot saved as elbow_plot.png")
    plt.close()

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    print("\nFIRST 10 CUSTOMERS WITH CLUSTERS")
    print(rfm.head(10))

    plt.figure(figsize=(8, 6))
    plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], cmap="viridis")
    plt.title("Customer Segmentation")
    plt.xlabel("Recency")
    plt.ylabel("Monetary")
    plt.grid(True)
    plt.savefig("customer_segmentation_plot.png")
    print("\nCluster scatter plot saved as customer_segmentation_plot.png")
    plt.close()

    print("\n==========================================")
    print("       CUSTOMERS PER CLUSTER")
    print("==========================================")
    cluster_counts = rfm["Cluster"].value_counts().sort_index()
    print(cluster_counts)

    plt.figure(figsize=(8, 6))
    cluster_counts.plot(kind="bar")
    plt.title("Number of Customers in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.grid(True)
    plt.savefig("cluster_distribution.png")
    print("\nCluster distribution plot saved as cluster_distribution.png")
    plt.close()

    print("\n==========================================")
    print("       CLUSTER PROFILE")
    print("==========================================")
    cluster_profile = rfm.groupby("Cluster").mean()
    print(cluster_profile)

    print("\n==========================================")
    print("     MARKETING RECOMMENDATIONS")
    print("==========================================")

    print("\nCluster 0:")
    print("Reward loyal customers with exclusive offers.")

    print("\nCluster 1:")
    print("Re-engage inactive customers using discounts and email campaigns.")

    print("\nCluster 2:")
    print("Promote premium products to high-spending customers.")

    print("\nCluster 3:")
    print("Encourage repeat purchases with personalized recommendations.")


if __name__ == "__main__":
    main()