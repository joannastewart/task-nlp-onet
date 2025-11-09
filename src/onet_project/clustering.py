import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score


#run PCA on SBERT embeddings to reduce dimensions to support clustering
def pca_on_embeddings(embeddings, n_components, random_state):
    pca = PCA(n_components=n_components, random_state=random_state)
    X = pca.fit_transform(embeddings)
    return X, pca

#assess k means on different k values using different evaluation metrics
def evaluate_kmeans(X, k_list, random_state):
    rows = []
    for k in k_list:
        km = KMeans(n_clusters=k,random_state=random_state,n_init="auto",)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        rows.append({"k": k,"silhouette": sil,"calinski_harabasz": ch,"davies_bouldin": db,"inertia": km.inertia_})

    df = pd.DataFrame(rows).sort_values("k")
    return df

def final_kmeans(X, k, random_state):
    km = KMeans(n_clusters=k,random_state=random_state,n_init="auto")
    labels = km.fit_predict(X)
    return labels, km