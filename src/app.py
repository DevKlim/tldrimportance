import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import uvicorn
import os

# ---- App and Model Configuration ----
app = FastAPI()

# Mount static files (for CSS) and templates (for HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Load Model and Tokenizer ---
# This assumes you have run train.py and the model is saved in ./results
MODEL_DIR = "./results"
tokenizer, model, device = None, None, None

@app.on_event("startup")
def load_model():
    """Load the model and tokenizer at application startup."""
    global tokenizer, model, device
    
    if not os.path.isdir(MODEL_DIR):
        print(f"Model directory '{MODEL_DIR}' not found.")
        print("Please train the model first by running: python src/train.py")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded successfully from {MODEL_DIR} on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")

# --- Prediction Logic ---
def predict_fluff(text: str) -> List[Dict]:
    """
    Takes a string of text and returns a list of tokens with their predicted labels.
    """
    if not model or not tokenizer:
        return [{"token": "Model not loaded.", "label": "FLUFF"}]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_labels_ids = predictions[0].cpu().tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    # Map ID to label name (e.g., 0 -> "FLUFF", 1 -> "ESSENTIAL")
    id2label = model.config.id2label
    result = []
    for token, label_id in zip(tokens, predicted_labels_ids):
        # We don't want to show special tokens like [CLS] or [SEP] in the output
        if token not in [tokenizer.cls_token, tokenizer.sep_token]:
            result.append({"token": token, "label": id2label[label_id]})
            
    return result

# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=Dict[str, List[Dict]])
async def handle_predict(text: str = Form(...)):
    """
    Handles the prediction request from the frontend.
    Receives text, runs prediction, and returns JSON.
    """
    predictions = predict_fluff(text)
    return {"predictions": predictions}

if __name__ == "__main__":
    # This block allows running the app directly for development
    # Use the `load_model` function before starting the server
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)