from src.onet_project.clustering import pca_on_embeddings,evaluate_kmeans,final_kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

#load data at task_text level
TASK_LEVEL_EMB_PATH = "../artifacts/embeddings/sbert_embeddings_task_level.npy"
TASK_LEVEL_DF_PATH = "../data/processed/analysis_task_level.parquet"
ANALYSIS_PATH = "../data/processed/analysis_df.parquet"
OUT_KMEANS_PATH = "../artifacts/reports/kmeans_eval.csv"
OUT_TASK_DF_PATH = "../data/processed/analysis_task_level_with_clusters.parquet"
OUT_DF_PATH = "../data/processed/analysis_df_with_clusters.parquet"
OUT_TASK_CSV_PATH = "../artifacts/reports/task_level_clusters_with_goodness.csv"


#load
task_embeddings = np.load(TASK_LEVEL_EMB_PATH)
task_text_df = pd.read_parquet(TASK_LEVEL_DF_PATH)
print("Task-level embeddings:", task_embeddings.shape)
print("Task-level df rows:   ", len(task_text_df))

random_state = 6740

#play with more and less dimension reduction using PCA
for n_comp in [20, 50, 100, 200]:
    X, _ = pca_on_embeddings(task_embeddings, n_components=n_comp, random_state=random_state)
    results = evaluate_kmeans(X, [200, 400], random_state=random_state)
    print(f"PCA={n_comp}\n", results)

#now finalize clustering
#first check variance explained with proposed
n_components = 20
pca = PCA(n_components=n_components)
pca.fit(task_embeddings)
print("pca explained variance",sum(pca.explained_variance_ratio_))
X, pca = pca_on_embeddings(task_embeddings, n_components=n_components, random_state=random_state)

k_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results_df = evaluate_kmeans(X, k_list, random_state=random_state)
print("KMeans evaluation:")
print(results_df)
results_df.to_csv(OUT_KMEANS_PATH, index=False)

k=300
labels,km = final_kmeans(X,k , random_state=random_state)

#calculate silhouette and dist to centroid at task level in pca space
sil = silhouette_samples(X, labels)
task_text_df["silhouette"] = sil
centers = km.cluster_centers_
dists = np.linalg.norm(X - centers[labels], axis=1)
task_text_df["dist_to_centroid"] = dists

#put cluster labels in task-level and dwa-task-level datasets
task_text_df["task_cluster_id"] = labels
analysis_df = pd.read_parquet(ANALYSIS_PATH)
analysis_with_clusters = analysis_df.merge(task_text_df[["task_id", "task_cluster_id"]],on="task_id",how="left")

task_text_df.to_parquet(OUT_TASK_DF_PATH,index=False)
analysis_with_clusters.to_parquet(OUT_DF_PATH,index=False)
task_text_df[["task_id", "task_text", "task_cluster_id", "silhouette", "dist_to_centroid"]].to_csv(OUT_TASK_CSV_PATH,index=False)

print("columns of df:", analysis_with_clusters.columns)

#plot k means silhouette vs k
plt.figure(figsize=(6, 4))
plt.plot(results_df["k"], results_df["silhouette"], marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score (higher is better)")
plt.title("KMeans Silhouette vs Number of Clusters")
plt.grid(True)
plt.tight_layout()
plt.savefig("../artifacts/figures/kmeans_silhouette_over_k.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(results_df["k"], results_df["davies_bouldin"], marker="o", color="green")
plt.xlabel("Number of clusters")
plt.ylabel("Davies–Bouldin (lower is better)")
plt.title("KMeans Davies–Bouldin vs Number of Clusters")
plt.grid(True)
plt.tight_layout()
plt.savefig("../artifacts/figures/kmeans_db_over_k.png", dpi=300)
plt.close()