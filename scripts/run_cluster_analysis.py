import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "../data/processed/analysis_df_with_clusters.parquet"
df = pd.read_parquet(data_path)

print("Rows:", len(df))
print("Unique clusters:", df["task_cluster_id"].nunique())
print("Unique DWAs:", df["dwa_id"].nunique())

#some eda on the new kmeans clusters
cluster_counts = df["task_cluster_id"].value_counts().rename_axis("task_cluster_id").reset_index(name="n_tasks")

print("\nTop 10 largest clusters:")
print(cluster_counts.head(10))
print("\nSmallest clusters:")
print(cluster_counts.tail(10))

#summarize each cluster by DWA composition

#count num of dwas
cluster_dwa_counts = (df.groupby("task_cluster_id",observed=True)["dwa_id"].nunique().reset_index(name="n_unique_dwas"))
#merge into cluster summary
cluster_summary = cluster_counts.merge(cluster_dwa_counts, on="task_cluster_id")
#what is top dwa and what proportion does it represent
dominant_dwa = (df.groupby(["task_cluster_id", "dwa_id"],observed=True).size().reset_index(name="count"))
dominant_dwa = dominant_dwa.loc[dominant_dwa.groupby("task_cluster_id",observed=True)["count"].idxmax()].rename(columns={"dwa_id": "dominant_dwa_id", "count": "dominant_dwa_count"})
cluster_summary = cluster_summary.merge(dominant_dwa, on="task_cluster_id")
cluster_summary["dominant_dwa_share"] = cluster_summary["dominant_dwa_count"] / cluster_summary["n_tasks"]
cluster_summary.sort_values("dominant_dwa_share", ascending=False, inplace=True)

print("\nCluster summary sample:")
print(cluster_summary.head(10))
cluster_summary.to_csv("../artifacts/reports/cluster_summary_vs_dwa.csv", index=False)

avg_dwas_per_cluster = cluster_summary["n_unique_dwas"].mean()
print(f"Average DWAs per cluster: {avg_dwas_per_cluster:.2f}")

#now do same analysis on dwas
dwa_counts = df["dwa_id"].value_counts().rename_axis("dwa_id").reset_index(name="n_tasks")

print("\nTop 10 largest DWAs:")
print(dwa_counts.head(10))
print("\nSmallest DWAs:")
print(dwa_counts.tail(10))

#summarize each DWA by task cluster composition

#count num of task clusters
dwa_cluster_counts = (df.groupby("dwa_id",observed=True)["task_cluster_id"].nunique().reset_index(name="n_unique_clusters"))
#merge into cluster summary
dwa_summary = dwa_counts.merge(dwa_cluster_counts, on="dwa_id")
#what is top cluster and what proportion does it represent
dominant_cluster = (df.groupby(["dwa_id","task_cluster_id"],observed=True).size().reset_index(name="count"))
dominant_cluster = dominant_cluster.loc[dominant_cluster.groupby("dwa_id",observed=True)["count"].idxmax()].rename(columns={"task_cluster_id": "dominant_cluster_id", "count": "dominant_cluster_count"})
dwa_summary = dwa_summary.merge(dominant_cluster, on="dwa_id")
dwa_summary["dominant_cluster_share"] = dwa_summary["dominant_cluster_count"] / dwa_summary["n_tasks"]
dwa_summary.sort_values("dominant_cluster_share", ascending=False, inplace=True)

print("\nDWA summary sample:")
print(dwa_summary.head(10))
dwa_summary.to_csv("../artifacts/reports/dwa_summary_vs_cluster.csv", index=False)

avg_clusters_per_dwa = dwa_summary["n_unique_clusters"].mean()
print(f"Average Clusters per DWA: {avg_clusters_per_dwa:.2f}")

#purity by dwa share
cluster_purity = np.average(cluster_summary["dominant_dwa_share"], weights=cluster_summary["n_tasks"])
print(f"\nWeighted average cluster purity (by dominant DWA): {cluster_purity:.3f}")

#purity by cluster share
dwa_purity = np.average(dwa_summary["dominant_cluster_share"], weights=dwa_summary["n_tasks"])
print(f"\nWeighted average dwa purity (by dominant cluster): {dwa_purity:.3f}")

#plot histogram of purity
plt.figure(figsize=(7, 4))
plt.hist(dwa_summary["dominant_cluster_share"], bins=20, edgecolor="black")
plt.xlabel("Share of DWA tasks in dominant semantic cluster")
plt.ylabel("Number of DWAs")
plt.title("Distribution of DWA Semantic Purity")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../artifacts/figures/dwa_purity.png", dpi=300)
plt.close()

#plot heatmap of cross-tab
#use a subset of DWAs for readability
top_n_dwas = 25

top_dwa_ids = (df["dwa_id"].value_counts().head(top_n_dwas).index)
cross = pd.crosstab(df["task_cluster_id"], df["dwa_id"])
cross_top = cross[top_dwa_ids]
cross_log = np.log1p(cross_top)  #log(1 + count) for better range in chart
plt.figure(figsize=(10, 6))
im = plt.imshow(cross_log, aspect="auto", interpolation="nearest")
plt.colorbar(im, label="log(1 + task count)")
plt.xlabel("DWA ID (top by frequency)")
plt.ylabel("Task cluster ID")
plt.title("Cluster × DWA Heatmap (Top DWAs)")
plt.tight_layout()
plt.savefig("../artifacts/figures/dwa_cluster_heatmap.png", dpi=300)
plt.close()