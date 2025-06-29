# TLDR Live Typing Highlighter

This project trains a machine learning model to provide real-time feedback on text to help users write more concise, "TLDR-style" summaries. As a user types into a textbox, words or phrases that the model considers "fluff" or non-essential are highlighted in red.

## How it Works

1.  **Data Preprocessing**: The project uses the `trl-lib/tldr` dataset. A one-time preprocessing script (`src/preprocess_data.py`) runs first. It uses `spaCy` to analyze each post and its summary. It identifies key phrases (like noun chunks) in the post that are semantically similar to the summary. These are considered "essential". This process generates a new labeled dataset saved as a CSV.

2.  **Modeling**: The problem is framed as a **Token Classification** task. We fine-tune a `DistilBERT` model on the preprocessed data to classify each word (token) from the input text as either `ESSENTIAL` or `FLUFF`.

3.  **Application**: A FastAPI backend serves the fine-tuned model. A simple HTML/JS frontend sends the user's text to the backend as they type. The backend returns chunks of the original text labeled as `FLUFF` or `ESSENTIAL`, which the frontend then uses to apply highlighting.

---

## How to Run with Docker Compose (GPU Recommended)

### Prerequisites for GPU Support

This is **CRITICAL**. The following must be installed on your host machine (your computer, not in Docker):
1.  **An NVIDIA GPU**.
2.  The latest **NVIDIA Drivers** for your GPU and OS.
3.  The **NVIDIA Container Toolkit**, which allows Docker to access your GPU. Follow the official installation guide for your Linux distribution.

### Running the Application

1.  **Build and Start the Services:**
    Open your terminal in the project root and run:
    ```bash
    docker-compose up --build