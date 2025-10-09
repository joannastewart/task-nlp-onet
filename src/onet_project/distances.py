import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import joblib

#load analysis file
def load_analysis_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["task_id"] = df["task_id"].astype(str)
    if df["dwa_id"].isna().any():
        missing = int(df["dwa_id"].isna().sum())
        raise ValueError(f"{missing} rows missing dwa_id.")
    return df

#load SBERT embeddings and normalize to unit length rows matched up to task ids
def load_sbert_aligned(df: pd.DataFrame, emb_npy: str, ids_parquet: str) -> np.ndarray:
    V_all = np.load(emb_npy, mmap_mode="r")
    ids = pd.read_parquet(ids_parquet)
    ids["task_id"] = ids["task_id"].astype(str)
    assert len(ids) == V_all.shape[0], "SBERT ids vs matrix row mismatch."
    pos = pd.Series(range(len(ids)), index=ids["task_id"])
    idx = pos.reindex(df["task_id"])
    if idx.isna().any():
        missing = df.loc[idx.isna(), "task_id"].head(5).tolist()
        raise ValueError(f"task_ids missing from SBERT ids, e.g. {missing}")
    V = V_all[idx.astype(int).to_numpy()].astype("float32", copy=False)
    return normalize(V, norm="l2", axis=1, copy=False)  #unit length rows

#transform task using pre-built tfidf vectorizer, normalize by row
def build_tfidf_matrix(df: pd.DataFrame, vect_path: str):
    vec = joblib.load(vect_path)
    X = vec.transform(df["task_text"].fillna("").astype(str))
    return normalize(X, norm="l2", axis=1, copy=False)

#extract row indices by dwa id
def group_indices(labels: pd.Series) -> dict:
    d = {}
    for i, lab in enumerate(labels):
        d.setdefault(lab, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in d.items()}

#calc centroid by dwa id for dense matrix, normalize
def dense_centroids_unit(V: np.ndarray, grp: dict) -> (np.ndarray, list):
    dwa_ids = list(grp.keys())
    C = np.zeros((len(dwa_ids), V.shape[1]), dtype="float32")
    for i, dwa in enumerate(dwa_ids):
        c = V[grp[dwa]].mean(axis=0)
        C[i] = c / (np.linalg.norm(c) + 1e-12)
    return C, dwa_ids

#for each DWA average its TF-IDF rows, create mean vector, re-normalize to unit length
def sparse_centroids_unit(X: sp.csr_matrix, grp: dict) -> (sp.csr_matrix, list):
    rows, cols, data= [], [], []
    dwa_ids = list(grp.keys())
    for i, dwa in enumerate(dwa_ids):
        idx = grp[dwa]
        coo = sp.coo_matrix(X[idx].sum(axis=0))
        cnt = max(len(idx), 1)
        rows.extend(np.zeros_like(coo.data, dtype=int) + i)
        cols.extend(coo.col)
        data.extend((coo.data / cnt).astype("float32"))
    C = sp.csr_matrix((data, (rows, cols)), shape=(len(dwa_ids), X.shape[1]), dtype="float32")
    row_norm = np.sqrt(C.multiply(C).sum(axis=1)).A.ravel() + 1e-12
    C = sp.diags((1.0 / row_norm).astype("float32")) @ C
    return C, dwa_ids

#for each row, calc cosine distance (dot product) from centroid
#cosine distance is 1-cosine similarity
def cosine_dist_dense(V: np.ndarray, C: np.ndarray, labels: pd.Series, dwa_ids: list) -> np.ndarray:
    pos = {d: i for i, d in enumerate(dwa_ids)}
    idx = np.array([pos[d] for d in labels], dtype=int)
    sims = np.einsum("ij,ij->i", V, C[idx]) #for each row i, multiply across columns j and sum
    return (1.0 - sims).astype("float32")

def cosine_dist_sparse(X: sp.csr_matrix, C: sp.csr_matrix, labels: pd.Series, dwa_ids: list) -> np.ndarray:
    pos = {d: i for i, d in enumerate(dwa_ids)}
    idx = np.array([pos[d] for d in labels], dtype=int)
    sims = np.array(X.multiply(C[idx]).sum(axis=1)).ravel()
    return (1.0 - sims).astype("float32")

#function to add z scores and other dwa level summary stats to df
def add_group_stats(df: pd.DataFrame, dist_col: str, prefix: str):
    g = df.groupby("dwa_id",observed=True)[dist_col]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    med = g.transform("median")
    df[f"z_{prefix}"]  = ((df[dist_col] - mu) / sd).fillna(0.0).astype("float32")
    df[f"pct_{prefix}"] = g.rank(pct=True, method="average").astype("float32")
    df[f"{prefix}_mean_dwa"] = mu.astype("float32")
    df[f"{prefix}_std_dwa"]  = sd.fillna(0.0).astype("float32")
    df[f"{prefix}_med_dwa"]  = med.astype("float32")
    return df