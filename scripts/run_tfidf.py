from src.onet_project.tfidf import calc_tfidf
import pandas as pd
import joblib
from scipy.sparse import save_npz

df = pd.read_parquet("../data/processed/analysis_df.parquet", engine="pyarrow")
X,vec = calc_tfidf(df['task_text'])
print("TF-IDF shape:", X.shape)
print("TF-IDF type:", type(X))
print("a few ngrams",vec.get_feature_names_out()[:10])
print("vec type",type(vec))
save_npz("../artifacts/embeddings/tfidf_X.npz", X)
joblib.dump(vec, "../artifacts/embeddings/tfidf_vectorizer.joblib")