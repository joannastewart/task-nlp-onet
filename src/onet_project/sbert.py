from sentence_transformers import SentenceTransformer

def calc_sbert(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)
