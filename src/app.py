import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
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
        print(f"Model directory '{MODEL_DIR}' not found. Please train the model first.")
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
def group_tokens_by_label(tokens, labels, offsets, original_text):
    """
    Groups consecutive tokens with the same label into single chunks,
    recovering the original text for each chunk using offsets.
    This handles subword tokens and reconstructs the original spacing.
    """
    results = []
    if not tokens:
        return results

    current_label = labels[0]
    start_offset = offsets[0][0]

    for i in range(1, len(tokens)):
        if labels[i] != current_label:
            end_offset = offsets[i - 1][1]
            results.append({
                "text": original_text[start_offset:end_offset],
                "label": current_label,
            })
            current_label = labels[i]
            start_offset = offsets[i][0]
    
    # Add the last chunk
    end_offset = offsets[-1][1]
    results.append({
        "text": original_text[start_offset:end_offset],
        "label": current_label,
    })
    
    return results

def predict_fluff(text: str) -> List[Dict]:
    """
    Takes a string of text and returns a list of text chunks with their predicted labels.
    This is robust to spacing and punctuation by using offset mapping.
    """
    if not model or not tokenizer or not text.strip():
        return [{"text": text, "label": "FLUFF"}] # Indicate model not loaded or handle empty/whitespace-only input

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        return_offsets_mapping=True
    )
    
    # offset_mapping gives (start_char, end_char) for each token
    offsets = inputs.pop("offset_mapping").squeeze().tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move all input tensors to device

    with torch.no_grad():
        logits = model(**inputs).logits # Pass all inputs to the model

    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    
    # Map ID to label name (e.g., 0 -> "FLUFF", 1 -> "ESSENTIAL")
    id2label = model.config.id2label

    # Filter out special tokens ([CLS], [SEP], [PAD]) and their corresponding offsets/predictions
    # Special tokens usually have (0,0) offsets
    valid_indices = [i for i, offset in enumerate(offsets) if offset != (0, 0)]
    
    filtered_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    filtered_tokens = [filtered_tokens[i] for i in valid_indices]
    filtered_labels = [id2label[predictions[i]] for i in valid_indices]
    filtered_offsets = [offsets[i] for i in valid_indices]

    if not filtered_tokens:
        # If after filtering, no valid tokens remain (e.g., input was just special chars)
        return [{"text": text, "label": "FLUFF"}]

    return group_tokens_by_label(filtered_tokens, filtered_labels, filtered_offsets, text)

# --- API Routes ---
@app.get("/", response_class=templates.TemplateResponse)
async def get_home(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def handle_predict(text: str = Form(...)):
    """
    Handles the prediction request from the frontend.
    Receives text, runs prediction, and returns structured JSON.
    """
    predictions = predict_fluff(text)
    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    # The load_model() call here is for local development when running app.py directly.
    # For production (e.g., with Gunicorn), the @app.on_event("startup") decorator handles it.
    load_model() 
    uvicorn.run(app, host="0.0.0.0", port=8000)