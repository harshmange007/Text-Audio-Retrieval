from collections import Counter
from sklearn.metrics import confusion_matrix

# FAISS-based query functions
from query.query_prototype import query as proto_query
from query.query_linear_projection import query as proj_query
from query.query_contrastive import query as contrastive_query

LABEL_MAP = {"drums": 0, "keys": 1}

EVAL_QUERIES = [
    ("drum samples", "drums"),
    ("drum loop", "drums"),
    ("percussion beat", "drums"),
    ("kick snare rhythm", "drums"),
    ("drums only", "drums"),

    ("keys melody", "keys"),
    ("piano chords", "keys"),
    ("keyboard harmony", "keys"),
    ("soft piano music", "keys"),
    ("keys loop", "keys"),
]

def predict_class(query_fn, text, top_k=5):
    results = query_fn(text, top_k=top_k)
    classes = [r["class"] for r in results]
    return Counter(classes).most_common(1)[0][0]

def evaluate_model(query_fn, model_name):
    y_true, y_pred = [], []

    for text, true_class in EVAL_QUERIES:
        pred_class = predict_class(query_fn, text)
        y_true.append(LABEL_MAP[true_class])
        y_pred.append(LABEL_MAP[pred_class])

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 55)
    print(f"{model_name} — FAISS-based Confusion Matrix")
    print("=" * 55)
    print("Rows: Actual | Columns: Predicted")
    print("        drums  keys")
    print(f"drums   {cm[0][0]:<6} {cm[0][1]}")
    print(f"keys    {cm[1][0]:<6} {cm[1][1]}")
    print("=" * 55)

    return cm

def main():
    evaluate_model(proto_query, "Prototype-Based Alignment")
    evaluate_model(proj_query, "Linear Projection Alignment")
    evaluate_model(contrastive_query, "Contrastive (Mini-CLAP) Alignment")

if __name__ == "__main__":
    main()
