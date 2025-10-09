import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_basics(df: pd.DataFrame) -> dict:
    tasks_per_dwa = (df.groupby("dwa_id",observed=True)["task_text"]
                       .nunique().sort_values(ascending=False))
    dwas_per_task = (df.groupby("task_text",observed=True)["dwa_id"]
                       .nunique().value_counts().sort_index())
    return {
        "n_rows": len(df),
        "n_tasks": df["task_text"].nunique(),
        "n_dwas": df["dwa_id"].nunique(),
        "tasks_per_dwa": tasks_per_dwa,
        "dwas_per_task_hist": dwas_per_task
    }

def plot_hist(series, title: str):
    fig, ax = plt.subplots(figsize=(6,4))
    series.plot(kind="hist", bins=20,ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

def top_ngrams(df,topk):
    agg = (df.groupby("dwa_id", observed=True)["task_text"]
           .apply(lambda s: " ".join(s.astype(str)))
           .sort_index())
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
        lowercase=True
    )
    X = vec.fit_transform(agg.values)
    terms = vec.get_feature_names_out()

    rows=[]
    for i, dwa in enumerate(agg.index):
        row = X.getrow(i)
        if row.nnz == 0:
            continue
        idx = row.toarray().ravel().argsort()[-topk:][::-1]
        for r, j in enumerate(idx, start=1):
            rows.append({"dwa_id": dwa, "term": terms[j],
                         "score": float(row[0, j]), "rank": r})

    out = pd.DataFrame(rows)

    titles = df[["dwa_id", "dwa_title"]].drop_duplicates("dwa_id")
    out = out.merge(titles, on="dwa_id", how="left")

    out = out.sort_values(["dwa_id", "rank"]).reset_index(drop=True)
    return out

