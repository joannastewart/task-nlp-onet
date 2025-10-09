from src.onet_project.sbert import calc_sbert
import pandas as pd
import numpy as np

#load existing analysis file
df = pd.read_parquet("../data/processed/analysis_df.parquet", engine="pyarrow")
#create embeddings as np
sbert_embeddings = calc_sbert(df['task_text']).astype("float32")

print("type:", type(sbert_embeddings))
print("SBERT shape:", sbert_embeddings.shape)
print("samples embedding",sbert_embeddings[:1,:10])

#save embeddings and task_ids separately
np.save("../artifacts/embeddings/sbert_embeddings.npy", sbert_embeddings)
df[['task_id']].to_parquet("../artifacts/embeddings/sbert_ids.parquet", index=False)