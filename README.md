# TLDR Live Typing Highlighter

This project trains a machine learning model to provide real-time feedback on text to help users write more concise, "TLDR-style" summaries. As a user types into a textbox, words or phrases that the model considers "fluff" or non-essential are highlighted in red.

## How it Works

1.  **Data**: The model is trained on the `trl-lib/tldr` dataset, which contains long Reddit posts and their corresponding TLDR summaries.
2.  **Modeling**: The problem is framed as a **Token Classification** task. We fine-tune a `DistilBERT` model to classify each word as either `ESSENTIAL` (likely to be in a TLDR) or `FLUFF` (unlikely to be in a TLDR).
3.  **Application**: A FastAPI backend serves the trained model. A simple HTML/JS frontend sends the user's text to the backend and highlights the response in real-time.

## Project Structure