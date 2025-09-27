import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def calc_tfidf(texts):
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english',min_df=3, max_df=0.8,
                          sublinear_tf=True,
                          lowercase=True,norm="l2",dtype=np.float32,analyzer="word")
    X = vec.fit_transform(texts) #returns sparse matrix where rows are texts and columns are n-grams
    feature_names = list(vec.get_feature_names_out())
    return X, feature_names