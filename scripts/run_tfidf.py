from src.onet_project.tfidf import calc_tfidf
import pandas as pd
from scipy.sparse import save_npz

df = pd.read_parquet("../data/processed/analysis_df.parquet", engine="pyarrow")
X,ngrams = calc_tfidf(df['task_text'])
print("TF-IDF shape:", X.shape)
print(ngrams[:5])
print(X[:5,:5])
#need to save these outputs