from src.onet_project.predict_DWA import (make_multilabel_dataset,
                                          build_ovr_log_pipeline,
                                          build_ovr_knn_pipeline,
                                          crossval_multilabel,
                                          random_prevalence_predictions)
import pandas as pd

df = pd.read_parquet("../data/processed/analysis_df.parquet", engine="pyarrow")

#create multi label dataset
texts, Y, mlb, task_ids, kept_DWAs = make_multilabel_dataset(df, min_per_dwa=20)
print("text shape:",len(texts))
print("text head:",texts.head(3))
print("Y shape:",Y.shape)
print("Y sample:",Y[:8,:8])
print("mlb dtype",type(mlb))
print("task ids shape:",task_ids.shape)
print("task ids sample",task_ids[:5])
print("kept_dwas shape",len(kept_DWAs))
print("kept_dwas sample",kept_DWAs[:5])

#run tfidf for logistic ovr
encoder = "tfidf"
enc_kwargs={}
pipe = build_ovr_log_pipeline(encoder=encoder, **enc_kwargs)
macro, micro, jaccard  = crossval_multilabel(pipe, texts, Y, task_ids, n_splits=5)
print(f"{encoder.upper()} LOGISTIC REGRESSION OvR — macro-F1: {macro:.3f} | micro-F1: {micro:.3f} | Jaccard: {jaccard:.3f}")


#run tfidf inside knn pipeline
encoder = "tfidf"
enc_kwargs={}
pipe = build_ovr_knn_pipeline(encoder=encoder, **enc_kwargs)
macro, micro, jaccard = crossval_multilabel(pipe, texts, Y, task_ids, n_splits=5)
print(f"{encoder.upper()} KNN OvR — macro-F1: {macro:.3f} | micro-F1: {micro:.3f} | Jaccard: {jaccard:.3f}")


#run sbert inside logistic pipeline
encoder = "sbert"
enc_kwargs=dict(
    model_name="all-MiniLM-L6-v2",
    max_len=128,
    batch_size=256,
    normalize=True)
pipe = build_ovr_log_pipeline(encoder=encoder, **enc_kwargs)
macro, micro, jaccard = crossval_multilabel(pipe, texts, Y, task_ids, n_splits=5)
print(f"{encoder.upper()} LOGISTIC REGRESSION OvR — macro-F1: {macro:.3f} | micro-F1: {micro:.3f} | Jaccard: {jaccard:.3f}")

#run sbert inside knn pipeline
encoder = "sbert"
enc_kwargs=dict(
    model_name="all-MiniLM-L6-v2",
    max_len=128,
    batch_size=256,
    normalize=True)
pipe = build_ovr_knn_pipeline(encoder=encoder, **enc_kwargs)
macro, micro, jaccard = crossval_multilabel(pipe, texts, Y, task_ids, n_splits=5)
print(f"{encoder.upper()} KNN OvR — macro-F1: {macro:.3f} | micro-F1: {micro:.3f} | Jaccard: {jaccard:.3f}")


#run random predictor maintaining prevalence by DWA
macro, micro, jaccard = random_prevalence_predictions(Y, seed = 6740)
print(f"Random Prevalance Predictions OvR — macro-F1: {macro:.3f} | micro-F1: {micro:.3f} | Jaccard: {jaccard:.3f}")