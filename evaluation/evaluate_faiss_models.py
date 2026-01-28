import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

from query.query_prototype import query as proto_query
from query.query_linear_projection import query as proj_query
from query.query_contrastive import query as contrastive_query

LABEL_MAP = {"drums": 0, "keys": 1}
LABELS = ["drums", "keys"]

EVAL_QUERIES = [
    # -------------------- DRUMS --------------------
    ("drum samples", "drums"),
    ("drum loop", "drums"),
    ("percussion beat", "drums"),
    ("kick snare rhythm", "drums"),
    ("drums only", "drums"),

    ("fast drum pattern", "drums"),
    ("electronic drum beat", "drums"),
    ("rhythmic percussion sounds", "drums"),
    ("heavy kick and snare", "drums"),
    ("tight drum groove", "drums"),

    # more complex / descriptive
    ("high energy drum loop for dance music", "drums"),
    ("percussive rhythm without melody", "drums"),
    ("looped drum pattern with strong beat", "drums"),
    ("drum sounds used for rhythm section", "drums"),
    ("short percussive hits and beats", "drums"),

    # intentionally challenging but still drums
    ("percussive elements driving the tempo", "drums"),
    ("rhythm focused loop with no harmonic content", "drums"),
    ("beat oriented audio sample", "drums"),

    # -------------------- KEYS --------------------
    ("keys melody", "keys"),
    ("piano chords", "keys"),
    ("keyboard harmony", "keys"),
    ("soft piano music", "keys"),
    ("keys loop", "keys"),

    ("melodic piano line", "keys"),
    ("harmonic keyboard progression", "keys"),
    ("smooth piano melody", "keys"),
    ("synth keys pad", "keys"),
    ("keyboard arpeggio", "keys"),

    # more complex / descriptive
    ("soft melodic piano chords", "keys"),
    ("keyboard based harmonic loop", "keys"),
    ("gentle keys progression with melody", "keys"),
    ("musical chord progression played on keys", "keys"),
    ("sustained keyboard tones with harmony", "keys"),

    # intentionally challenging but still keys
    ("melody driven musical loop", "keys"),
    ("harmonic content without strong rhythm", "keys"),
    ("tonal keyboard sounds for background music", "keys"),
]


def predict_class(query_fn, text, top_k=5):
    results = query_fn(text, top_k=top_k)
    classes = [r["class"] for r in results]
    return Counter(classes).most_common(1)[0][0]

def save_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(model_name)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

    filename = f"{model_name.lower().replace(' ', '_')}_faiss_cm.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"✅ Saved: {filename}")

def evaluate_model(query_fn, model_name):
    y_true, y_pred = [], []

    for text, true_class in EVAL_QUERIES:
        pred_class = predict_class(query_fn, text)
        y_true.append(LABEL_MAP[true_class])
        y_pred.append(LABEL_MAP[pred_class])

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, model_name)

def main():
    evaluate_model(proto_query, "Prototype")
    evaluate_model(proj_query, "Linear Projection")
    evaluate_model(contrastive_query, "Contrastive")

if __name__ == "__main__":
    main()
