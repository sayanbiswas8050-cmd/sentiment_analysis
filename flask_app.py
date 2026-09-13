import pickle
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# Preprocessing Import (with fallback)
# ---------------------------------------------------------
try:
    from preprocessing import LABEL, preprocess_comment
except ImportError:
    LABEL = {0: "negative", 1: "neutral", 2: "positive"}
    def preprocess_comment(text: str) -> str:
        return text.strip().lower()

# ---------------------------------------------------------
# App & Model Initialization
# ---------------------------------------------------------
app = Flask(__name__)

MODEL_DIR = Path(__file__).parent / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"

models = {}

def load_models():
    """Load TF-IDF vectorizer and LightGBM model from disk."""
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        print(f"[Warning] Missing model artifacts at:\n - {VECTORIZER_PATH}\n - {MODEL_PATH}")
        return

    try:
        with open(VECTORIZER_PATH, "rb") as f:
            models["vectorizer"] = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            models["model"] = pickle.load(f)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")

load_models()

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # If opened in browser via GET, render index.html from templates/
    if request.method == "GET":
        return render_template("index.html")

    # If POST, process the prediction
    if "vectorizer" not in models or "model" not in models:
        return jsonify({"error": "Models are not loaded or missing from models/ folder"}), 503

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing or empty 'text' field in request body"}), 400

    cleaned = preprocess_comment(text)
    vectorizer = models["vectorizer"]
    model = models["model"]

    x = vectorizer.transform([cleaned])
    pred_label = int(model.predict(x)[0])
    sentiment = LABEL.get(pred_label, str(pred_label))

    return jsonify({
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "label": pred_label
    })

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "status": "ok",
        "model_loaded": "model" in models,
        "vectorizer_loaded": "vectorizer" in models
    })

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)