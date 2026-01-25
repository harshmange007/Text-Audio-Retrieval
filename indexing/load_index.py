import json
import faiss
import os

def load_faiss_index(index_name):
    base = "indexing/indices"
    index_path = os.path.join(base, f"{index_name}.faiss")
    meta_path = os.path.join(base, f"{index_name}_metadata.json")

    index = faiss.read_index(index_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return index, metadata
