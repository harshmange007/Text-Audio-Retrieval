from indexing.load_index import load_faiss_index
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index (audio-only)
index, metadata = load_faiss_index("prototype_audio_index")

# Text model (for class routing only)
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CLASS_TEXTS = ["drums", "keys"]

def query(text, top_k=5):
    # 1️⃣ Predict class from text
    class_embs = text_model.encode(CLASS_TEXTS, normalize_embeddings=True)
    q = text_model.encode([text], normalize_embeddings=True)

    label = int(np.argmax(q @ class_embs.T))
    label_name = CLASS_TEXTS[label]

    # 2️⃣ Retrieve audio samples belonging to that class
    results = []
    for m in metadata:
        if m["class"] == label_name:
            results.append(m)
        if len(results) == top_k:
            break

    return results

if __name__ == "__main__":
    print(query("drum loop"))
    print(query("piano melody"))

