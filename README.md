# TLDR Live Typing Highlighter

This project trains a machine learning model to provide real-time feedback on text to help users write more concise, "TLDR-style" summaries. As a user types into a textbox, words or phrases that the model considers "fluff" or non-essential are highlighted in red.

## How it Works

1.  **Data**: The model is trained on the `trl-lib/tldr` dataset.
2.  **Modeling**: The problem is framed as a **Token Classification** task. We fine-tune a `DistilBERT` model to classify each word as either `ESSENTIAL` or `FLUFF`.
3.  **Application**: A FastAPI backend serves the trained model. A simple HTML/JS frontend sends the user's text to the backend and highlights the response in real-time.

---

## How to Run with Docker Compose (GPU Recommended)

This is the easiest and most powerful way to run the project. It automates the entire workflow: training on the GPU and then serving the model.

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