import numpy as np
from sentence_transformers import SentenceTransformer
from indexing.load_index import load_faiss_index

# Load FAISS
index, metadata = load_faiss_index("contrastive_audio_index")

# Load text model
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load projection weights
W = np.load("embeddings/projected/contrastive_text_W.npy")
b = np.load("embeddings/projected/contrastive_text_b.npy")

def project_text(x):
    x = x @ W.T + b
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def query(text, top_k=5):
    text_emb = text_model.encode([text], normalize_embeddings=True)
    text_proj = project_text(text_emb.astype("float32"))
    scores, idxs = index.search(text_proj, top_k)
    return [metadata[i] for i in idxs[0]]

if __name__ == "__main__":
    print(query("give me drum samples"))
    print(query("soft piano chords"))
