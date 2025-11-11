import numpy as np
from src.onet_project.distances import (
      load_analysis_df,
      load_sbert_aligned,
      build_tfidf_matrix,
      group_indices,
      dense_centroids_unit,
      sparse_centroids_unit,
      cosine_dist_sparse,
      cosine_dist_dense,
      add_group_stats,
      top_dense,
      top_sparse
)
#paths for the different files to load
ANALYSIS_PATH = "../data/processed/analysis_df.parquet"
SBERT_NPY_PATH = "../artifacts/embeddings/sbert_embeddings.npy"
SBERT_IDS_PATH = "../artifacts/embeddings/sbert_ids.parquet"
TFIDF_VECT_PATH = "../artifacts/embeddings/tfidf_vectorizer.joblib"
OUT_PATH = "../data/processed/analysis_with_distances.parquet"

df = load_analysis_df(ANALYSIS_PATH)
print(f"Loaded analysis_df: {df.shape}")

V = load_sbert_aligned(df, SBERT_NPY_PATH, SBERT_IDS_PATH)
print(f"Aligned SBERT matrix: {V.shape}")

X = build_tfidf_matrix(df, TFIDF_VECT_PATH)
print(f"Built TF-IDF matrix: {X.shape}")

grp = group_indices(df["dwa_id"])

C_sbert, dwa_ids = dense_centroids_unit(V, grp)
C_tfidf, _       = sparse_centroids_unit(X, grp)
print(f"Centroids — SBERT: {C_sbert.shape}, TF-IDF: {C_tfidf.shape}")

#distances are 1-cos similarity
df["dist_sbert"] = cosine_dist_dense(V, C_sbert, df["dwa_id"], dwa_ids)
df["dist_tfidf"] = cosine_dist_sparse(X, C_tfidf, df["dwa_id"], dwa_ids)

print("Distance ranges — "
      f"SBERT: [{df['dist_sbert'].min():.3f}, {df['dist_sbert'].max():.3f}]  "
      f"TF-IDF: [{df['dist_tfidf'].min():.3f}, {df['dist_tfidf'].max():.3f}]")

#add count of tasks per dwa
df["tasks_per_dwa"] = df.groupby("dwa_id",observed=True)["task_id"].transform("size").astype("int32")

df = add_group_stats(df, "dist_sbert",  "sbert")
df = add_group_stats(df, "dist_tfidf",  "tfidf")

print("Z score ranges — "
      f"SBERT: [{df['z_sbert'].min():.3f}, {df['z_sbert'].max():.3f}]  "
      f"TF-IDF: [{df['z_tfidf'].min():.3f}, {df['z_tfidf'].max():.3f}]")

#add closest centroid for each task for each type of distance
dwa_ids_arr = np.asarray(dwa_ids)
#SBERT closest
best_idx_s, best_dist_s = top_dense(V, C_sbert)
df["closest_dwa_sbert"] = dwa_ids_arr[best_idx_s]
df["closest_dist_sbert"] = best_dist_s.astype("float32")
df["mismatch_sbert"] = (df["closest_dwa_sbert"] != df["dwa_id"])

#TF-IDF closest
best_idx_t, best_dist_t = top_sparse(X, C_tfidf)
df["closest_dwa_tfidf"] = dwa_ids_arr[best_idx_t]
df["closest_dist_tfidf"] = best_dist_t.astype("float32")
df["mismatch_tfidf"] = (df["closest_dwa_tfidf"] != df["dwa_id"])

print("mismatch_sbert:", int(df["mismatch_sbert"].sum()), f"({df['mismatch_sbert'].mean():.1%})")
print("mismatch_tfidf:", int(df["mismatch_tfidf"].sum()), f"({df['mismatch_tfidf'].mean():.1%})")

#get dwa_title for closest centroid
id_to_title = (df[["dwa_id","dwa_title"]].dropna().drop_duplicates("dwa_id").set_index("dwa_id")["dwa_title"])
df["closest_dwatitle_sbert"] = df["closest_dwa_sbert"].map(id_to_title)
df["closest_dwatitle_tfidf"] = df["closest_dwa_tfidf"].map(id_to_title)
#print(df.head(10).T)

df.to_parquet(OUT_PATH, index=False)
