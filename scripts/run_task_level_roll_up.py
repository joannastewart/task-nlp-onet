import numpy as np
import pandas as pd

ANALYSIS_PATH = "../data/processed/analysis_df.parquet"
SBERT_NPY_PATH = "../artifacts/embeddings/sbert_embeddings.npy"
SBERT_IDS_PATH = "../artifacts/embeddings/sbert_ids.parquet"
OUT_DF_PATH = "../data/processed/analysis_task_level.parquet"
OUT_EMBED_PATH = "../artifacts/embeddings/sbert_embeddings_task_level.npy"

embeddings = np.load(SBERT_NPY_PATH)
ids = pd.read_parquet(SBERT_IDS_PATH)
analysis_df = pd.read_parquet(ANALYSIS_PATH)
assert len(ids) == len(embeddings)
assert len(ids) == len(analysis_df)

#will collapse to task_text level so let's preserve order
ids = ids.copy()
ids["emb_i"] = np.arange(len(ids))

#collapse to task_text level
task_level_ids = (ids.sort_values("emb_i").drop_duplicates(subset="task_id", keep="first").reset_index(drop=True))

#grab matching embeddings
task_embeddings = embeddings[task_level_ids["emb_i"].values]
print("task embeddings shape",task_embeddings.shape)

#grab the text it maps to (I think it's already deduped but making sure)
tasks_text = (analysis_df[["task_id", "task_text"]].drop_duplicates("task_id", keep="first"))
task_text_df = task_level_ids.merge(tasks_text, on="task_id", how="left")
print("tast level df length",len(task_text_df))
print(task_text_df.head(3))
#save embeddings and df
np.save(OUT_EMBED_PATH, task_embeddings)
task_text_df.to_parquet(OUT_DF_PATH, index=False)

