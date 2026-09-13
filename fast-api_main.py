import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from preprocessing import LABEL, preprocess_comment

MODEL_DIR = Path(__file__).parent / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifacts missing. Expected:\n"
            f" - {VECTORIZER_PATH}\n"
            f" - {MODEL_PATH}"
        )
    
    with open(VECTORIZER_PATH, "rb") as file:
        models["vectorizer"] = pickle.load(file)

    with open(MODEL_PATH, "rb") as file:
        models["model"] = pickle.load(file)

    yield
    models.clear()

app = FastAPI(
    title="Reddit Comment Sentiment API",
    description="TF-IDF + LightGBM sentiment classifier (negative / neutral / positive)",
    version="1.0.0",
    lifespan=lifespan,
)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw comment text to classify")

class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    label: int

def predict_sentiment(text: str) -> PredictResponse:

    vectorizer = models.get("vectorizer")
    model = models.get("model")

    if not vectorizer or not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready.",
        )

    cleaned = preprocess_comment(text)
    
    # LightGBM supports sparse matrices directly from vectorizer.transform
    x = vectorizer.transform([cleaned])
    
    pred_label = int(model.predict(x)[0])

    return PredictResponse(
        text=text,
        cleaned_text=cleaned,
        sentiment=LABEL.get(pred_label, str(pred_label)),
        label=pred_label,
    )

@app.get("/info")
def info():
    return {
        "status": "ok",
        "model_loaded": "model" in models,
        "vectorizer_loaded": "vectorizer" in models,
    }

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    return predict_sentiment(payload.text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)