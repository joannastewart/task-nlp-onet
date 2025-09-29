from collections import Counter
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, jaccard_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier



#first roll up df to task level. Will keep multiple DWAs per task in form of matrix.
def make_multilabel_dataset(df: pd.DataFrame, min_per_dwa: int = 12
) -> Tuple[pd.Series, np.ndarray, MultiLabelBinarizer, np.ndarray, List[str]]:
    """
    Collapse to 1 row per task_id with a set of DWAs; keep only frequent DWAs.
    Returns: texts, Y (multi-hot), mlb, groups (task_id), kept_classes
    """
    grouping = (df.groupby("task_id", observed=True)
            .agg(task_text=("task_text","first"),
                 labels=("dwa_id", lambda s: sorted(set(s)))))
    #creates task text and labels which is a set for all DWA_IDs
    #now count DWAs after flattening into a stream of labels; count by DWA
    cnt = Counter(l for L in grouping["labels"] for l in L)
    #now keep DWAs over the threshold task count
    kept = sorted([d for d, c in cnt.items() if c >= min_per_dwa])

    #filter out dwas below threshold
    grouping["labels"] = grouping["labels"].map(lambda L: [l for l in L if l in kept])
    #drop tasks with no kept DWAs
    grouping = grouping[grouping["labels"].map(len) > 0].copy()

    mlb = MultiLabelBinarizer(classes=kept)
    Y = mlb.fit_transform(grouping["labels"])

    #series of tasks belonging to DWAs that met threshold
    texts = grouping["task_text"]
    #create array with index of grouping DF which is task id; need this for modeling later
    groups = grouping.index.to_numpy()

    return texts, Y, mlb, groups, kept

#build embedder for tfidf
def make_tfidf():
    return TfidfVectorizer(
        ngram_range=(1,2), min_df=3, max_df=0.8,
        stop_words="english", sublinear_tf=True,
        lowercase=True,norm="l2",dtype=np.float32,analyzer="word"
    )

#build embedder for sbert

class SBERTEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 max_len: int = 128, batch_size: int = 256,
                 normalize: bool = True, device: Optional[str] = None):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None

    def fit(self, X: List[str], y=None):
        #lazy import so TF-IDF runs don’t require sentence-transformers installed
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X: List[str]): #these subclasses make it a scikitlearn transformer
        texts = [t if isinstance(t, str) else "" for t in X]
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        ).astype(np.float32)
        return emb

#build pipeline
def build_ovr_log_pipeline(encoder: str = "tfidf", **enc_kwargs) -> Pipeline:
    if encoder == "tfidf":
        enc = make_tfidf()
    elif encoder == "sbert":
        enc = SBERTEncoder(**enc_kwargs)
    else:
        raise ValueError("encoder must be 'tfidf' or 'sbert'")

    clf = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000),
        n_jobs=-1
    )
    return Pipeline([("embed", enc), ("clf", clf)])

def build_ovr_knn_pipeline(encoder: str = "tfidf", **enc_kwargs) -> Pipeline:
    if encoder == "tfidf":
        enc = make_tfidf()
    elif encoder == "sbert":
        enc = SBERTEncoder(**enc_kwargs)
    else:
        raise ValueError("encoder must be 'tfidf' or 'sbert'")
    knn = KNeighborsClassifier(
        n_neighbors=3, metric="cosine", algorithm="brute", weights="distance"
    )
    return Pipeline([("embed", enc), ("clf", knn)])

#evaluation
def crossval_multilabel(pipeline, texts, Y, task_ids, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    macro, micro, jaccard = [], [], []
    for train, test in gkf.split(texts, Y, task_ids):
        pipeline.fit(texts.iloc[train], Y[train])
        pred = pipeline.predict(texts.iloc[test])
        macro.append(f1_score(Y[test], pred, average="macro", zero_division=0))
        #macro computes F1 for each class then averages
        micro.append(f1_score(Y[test], pred, average="micro", zero_division=0))
        #micro pools across all and calculates once
        jaccard.append(jaccard_score(Y[test], pred, average="samples", zero_division=0))
        #jaccard is the overlap of true/predict divided by the total unique labels then averaged across samples
    return float(np.mean(macro)), float(np.mean(micro)), float(np.mean(jaccard))

def random_prevalence_predictions(Y_true, seed):
    rng = np.random.default_rng(seed)
    p = Y_true.mean(axis=0)
    Y_pred = (rng.random(Y_true.shape) < p).astype(int)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    jaccard = jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)
    return float(np.mean(macro_f1)), float(np.mean(micro_f1)), float(np.mean(jaccard))
