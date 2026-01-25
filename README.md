# 🎧 Text–Audio Retrieval System using Multimodal Embeddings

## 📌 Problem Statement

The goal of this project is to build a system that can **index audio samples and retrieve them using natural language queries**.  
Given a text prompt like:

> *“give me drum samples”*

the system should return **relevant audio samples** (e.g., drum loops) from a database.

Key requirements:
- Audio and text must be represented in a **common semantic space**
- Retrieval should be **promptable** (text → audio)
- Multiple modeling approaches should be explored and compared
- Evaluation should include **confusion matrices**
- A **vector database** should be used for indexing and retrieval

---

## 📊 Dataset

- **Source**: `splice.com`
- **Classes**:
  - `Drums`
  - `Keys`
- **Samples**:
  - 20 audio samples per class (40 total)
- **Collection**:
  - Downloaded manually using browser extensions
  - Stored locally as `.wav` files

---

## 🧠 Approaches Explored

We implemented and compared **three different alignment strategies** for text–audio retrieval.

---

### 🔹 Approach 1: Prototype-Based Alignment (Baseline)

**Idea**:
- Extract audio embeddings using a pretrained audio model
- Compute **class prototypes** (mean embeddings) for `drums` and `keys`
- Route text queries to a class using text embeddings
- Retrieve audio samples belonging to that class

**Characteristics**:
- No learning
- Simple and interpretable
- Serves as a strong baseline

---

### 🔹 Approach 2: Linear Projection using Class Labels

**Idea**:
- Embed audio samples using an audio embedding model
- Embed class text labels (`"drums"`, `"keys"`) using a text embedding model
- Learn a **linear projection** that maps audio embeddings into the text embedding space
- Train using **cosine similarity loss**

**Why this works**:
- Uses labels as a **semantic bridge**
- Aligns modalities without using CLAP

---

### 🔹 Approach 3: Contrastive Learning (Mini-CLAP, Built from Scratch)

**Idea**:
- Build a lightweight CLAP-like architecture
- Separate projection heads for audio and text
- Train using **contrastive (cosine) loss**
- Encourage matching text–audio pairs to be close, others far apart

**Constraints**:
- Very small dataset
- Implemented from scratch in limited time (no CLAP used)

---

## 🗂️ Vector Database (FAISS)

To make the system **scalable and production-like**, we use **FAISS** as a vector database.

- Audio embeddings are indexed using `IndexFlatIP`
- Cosine similarity achieved via L2 normalization
- Separate FAISS indices for each approach:
  - Prototype
  - Linear Projection
  - Contrastive

⚠️ On macOS (Apple Silicon), FAISS and PyTorch can conflict.  
To handle this safely:
- All **Torch operations are done offline**
- FAISS indexing and querying are **NumPy-only**

This mirrors real-world ML systems.

---

## 🔍 Retrieval Pipeline


Users can issue queries like:
- `"give me drum samples"`
- `"piano melody"`
- `"soft keyboard chords"`

---

## 📈 Evaluation & Results

Evaluation is performed using **FAISS-based retrieval** (not in-memory similarity).

- 10 evaluation queries (5 per class)
- Top-k retrieval with majority voting
- Confusion matrices saved as **PNG files in project root**

### Confusion Matrix (all three approaches)

| Actual \ Predicted | Drums | Keys |
|--------------------|-------|------|
| **Drums**          | 5     | 0    |
| **Keys**           | 1     | 4    |

### Metrics (identical for all approaches)

- **Accuracy**: 90%
- **Drums**
  - Precision: 83%
  - Recall: 100%
- **Keys**
  - Precision: 100%
  - Recall: 80%

📌 Interpretation:
- Drums are acoustically distinctive → perfect recall
- Some rhythmic keys samples are confused with drums
- With only two coarse classes and a small dataset, all methods converge to similar performance

---

## 📁 Project Structure

audio_text_retrieval/
│
├── data/
│   ├── drums/                  # Raw drum audio samples
│   └── keys/                   # Raw keys audio samples
│
├── preprocessing/
│   └── audio_preprocess.py     # Audio loading, resampling, normalization
│
├── embeddings/
│   ├── extract_audio_embeddings.py
│   ├── saved/                  # Raw audio embeddings, labels, filenames
│   └── projected/              # Projected embeddings (linear / contrastive)
│
├── models/
│   ├── prototype_based/
│   │   └── prototype_retrieval.py
│   ├── linear_projection/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── export_embeddings.py
│   └── contrastive/
│       ├── model.py
│       ├── train.py
│       ├── export_embeddings.py
│       └── export_text_projection.py
│
├── indexing/
│   ├── build_faiss_index.py    # Builds FAISS indices (NumPy-only)
│   └── load_index.py
│
├── query/
│   ├── query_prototype.py
│   ├── query_linear_projection.py
│   └── query_contrastive.py
│
├── evaluation/
│   └── evaluate_faiss_models.py
│
├── prototype_faiss_cm.png
├── linear_projection_faiss_cm.png
├── contrastive_faiss_cm.png
│
├── requirements.txt
└── README.md



---

## 🚀 Key Takeaways

- Built an **end-to-end multimodal retrieval system**
- Explored **three alignment strategies**
- Implemented **FAISS-based vector indexing**
- Designed a **promptable text → audio retrieval pipeline**
- Performed **fair, system-consistent evaluation**
- Carefully handled real-world engineering constraints (FAISS + macOS)

---

## 🔮 Future Work

- Add finer-grained classes (kick, snare, piano, synth, etc.)
- Increase dataset size
- Use richer text descriptions per audio sample
- Replace MiniLM with stronger text encoders
- Deploy as a web or chatbot interface
