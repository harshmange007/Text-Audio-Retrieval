from sentence_transformers import SentenceTransformer
from indexing.load_index import load_faiss_index

# Load FAISS
index, metadata = load_faiss_index("linear_projection_index")

# Text embedding model (SAFE)
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def query(text, top_k=5):
    text_emb = text_model.encode([text], normalize_embeddings=True)
    scores, idxs = index.search(text_emb.astype("float32"), top_k)
    return [metadata[i] for i in idxs[0]]

if __name__ == "__main__":
    print(query("give me drum samples"))
    print(query("piano melody"))
