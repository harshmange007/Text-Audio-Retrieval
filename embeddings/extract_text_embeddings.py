import os
import numpy as np
from sentence_transformers import SentenceTransformer

SAVE_DIR = "embeddings/saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load text embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Class labels (semantic anchors)
CLASS_TEXTS = [
    "drums",
    "keys"
]

def main():
    embeddings = model.encode(
        CLASS_TEXTS,
        normalize_embeddings=True
    )

    np.save(os.path.join(SAVE_DIR, "text_class_embeddings.npy"), embeddings)
    np.save(os.path.join(SAVE_DIR, "text_class_labels.npy"), np.array(CLASS_TEXTS))

    print("✅ Text class embeddings saved.")

if __name__ == "__main__":
    main()
