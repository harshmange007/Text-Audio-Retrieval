import os
import json
import numpy as np
import faiss
import sys

SAVE_DIR = "indexing/indices"
os.makedirs(SAVE_DIR, exist_ok=True)

def build_index(embeddings, metadata, index_name):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(SAVE_DIR, f"{index_name}.faiss")
    faiss.write_index(index, index_path)

    meta_path = os.path.join(SAVE_DIR, f"{index_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ FAISS index built: {index_name}")
    print(f"   vectors: {index.ntotal}, dim: {dim}")

def build_prototype_index():
    audio_emb = np.load("embeddings/saved/audio_embeddings.npy")
    labels = np.load("embeddings/saved/labels.npy")
    filenames = np.load("embeddings/saved/filenames.npy", allow_pickle=True)

    audio_emb = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)

    metadata = [
        {"filename": str(f), "class": "drums" if l == 0 else "keys"}
        for f, l in zip(filenames, labels)
    ]

    build_index(audio_emb, metadata, "prototype_audio_index")

def build_projection_index():
    emb = np.load("embeddings/projected/linear_projection_embeddings.npy")
    labels = np.load("embeddings/projected/labels.npy")
    filenames = np.load("embeddings/projected/filenames.npy", allow_pickle=True)

    metadata = [
        {"filename": str(f), "class": "drums" if l == 0 else "keys"}
        for f, l in zip(filenames, labels)
    ]

    build_index(emb, metadata, "linear_projection_index")

def build_contrastive_index():
    emb = np.load("embeddings/projected/contrastive_audio_embeddings.npy")
    labels = np.load("embeddings/projected/labels.npy")
    filenames = np.load("embeddings/projected/filenames.npy", allow_pickle=True)

    metadata = [
        {"filename": str(f), "class": "drums" if l == 0 else "keys"}
        for f, l in zip(filenames, labels)
    ]

    build_index(emb, metadata, "contrastive_audio_index")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m indexing.build_faiss_index [prototype|projection|contrastive]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "prototype":
        build_prototype_index()
    elif mode == "projection":
        build_projection_index()
    elif mode == "contrastive":
        build_contrastive_index()
    else:
        raise ValueError("Unknown mode")
