from collections import Counter
from sklearn.metrics import confusion_matrix

# ---- IMPORT RETRIEVAL FUNCTIONS ----
from models.prototype_based.prototype_retrieval import retrieve as retrieve_proto
from models.linear_projection.inference import retrieve as retrieve_proj
from models.contrastive.inference import retrieve as retrieve_contrastive

LABEL_MAP = {"drums": 0, "keys": 1}
INV_LABEL_MAP = {0: "drums", 1: "keys"}

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


def predict_class(retrieve_fn, query, top_k=5):
    results = retrieve_fn(query, top_k=top_k)
    classes = [cls for _, cls in results]
    return Counter(classes).most_common(1)[0][0]

def evaluate(retrieve_fn, name):
    y_true = []
    y_pred = []

    for query, true_class in EVAL_QUERIES:
        pred_class = predict_class(retrieve_fn, query)
        y_true.append(LABEL_MAP[true_class])
        y_pred.append(LABEL_MAP[pred_class])

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"{name} Confusion Matrix")
    print("=" * 50)
    print("Rows: Actual | Columns: Predicted")
    print("        drums  keys")
    print(f"drums   {cm[0][0]:<6} {cm[0][1]}")
    print(f"keys    {cm[1][0]:<6} {cm[1][1]}")
    print("=" * 50)

    return cm

def main():
    evaluate(retrieve_proto, "Prototype-Based Alignment")
    evaluate(retrieve_proj, "Linear Projection Alignment")
    evaluate(retrieve_contrastive, "Contrastive (Mini-CLAP) Alignment")

if __name__ == "__main__":
    main()
