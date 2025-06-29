import os
import json
import pandas as pd
import spacy
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import re

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
LABELS = ["FLUFF", "ESSENTIAL"]  # 0: FLUFF, 1: ESSENTIAL
OUTPUT_DIR = "/app/data"
PREPROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "tldr_preprocessed.csv")


# --- Part 1: Initial, one-time data generation ---

def create_preprocessed_csv():
    """
    Loads the raw TLDR dataset, processes it to find essential phrases using spaCy,
    and saves the result to a CSV file. This is a one-time, expensive operation.
    """
    if os.path.exists(PREPROCESSED_DATA_PATH):
        print(f"Preprocessed data already found at {PREPROCESSED_DATA_PATH}. Skipping.")
        return

    print("Loading 'en_core_web_md' spaCy model...")
    nlp = spacy.load("en_core_web_md", disable=["ner"])

    print("Loading raw 'trl-lib/tldr' dataset...")
    raw_dataset = load_dataset("trl-lib/tldr", split="train").shuffle(seed=42).select(range(20000))

    def parse_post_from_prompt(prompt_text):
        match = re.search(r"POST:\s*(.*)\s*TL;DR:", prompt_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    posts, summaries = [], []
    print("Parsing posts and summaries from dataset...")
    for example in tqdm(raw_dataset, desc="Extracting content"):
        post = parse_post_from_prompt(example['prompt'])
        summary = example['completion']
        if post and summary and len(post) > 50 and len(summary) > 10:
            posts.append(post)
            summaries.append(summary)

    # OPTIMIZATION: Use nlp.pipe for efficient batch processing.
    print("Processing summaries with spaCy...")
    summary_docs = list(tqdm(nlp.pipe(summaries, batch_size=50), total=len(summaries), desc="Processing Summaries"))
    print("Processing posts with spaCy...")
    post_docs = list(tqdm(nlp.pipe(posts, batch_size=50), total=len(posts), desc="Processing Posts"))

    processed_records = []
    print("Finding essential phrases by comparing posts to summaries...")
    for i in tqdm(range(len(post_docs)), desc="Analyzing Documents"):
        post_doc, summary_doc = post_docs[i], summary_docs[i]
        
        # Skip documents if they don't have vectors, as similarity won't work.
        if not post_doc.has_vector or not summary_doc.has_vector: 
            continue

        candidate_phrases = {}
        for chunk in post_doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if len(phrase) < 4: continue

            # Ensure vectors are available and not zero-norm before calling similarity.
            if chunk.has_vector and chunk.vector_norm and summary_doc.vector_norm:
                similarity_score = chunk.similarity(summary_doc)
                
                if similarity_score > 0.75:
                    candidate_phrases[phrase] = float(similarity_score)

        top_phrases = dict(sorted(candidate_phrases.items(), key=lambda item: item[1], reverse=True)[:30])

        if top_phrases:
            processed_records.append({
                "post": posts[i],
                "similar": json.dumps(top_phrases)
            })

    if not processed_records:
        raise RuntimeError("No records were processed. Check preprocessing logic and dataset.")

    print(f"Creating DataFrame from {len(processed_records)} records...")
    df = pd.DataFrame(processed_records)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving preprocessed data to {PREPROCESSED_DATA_PATH}...")
    df.to_csv(PREPROCESSED_DATA_PATH, index=False)
    print("Initial preprocessing complete.")


# --- Part 2: Function for train.py to load and tokenize data ---

def get_tokenized_datasets():
    """
    Loads the preprocessed CSV and prepares tokenized datasets for the Trainer.
    """
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Preprocessed data not found. Run this script directly to generate it.")

    print(f"Loading preprocessed dataset from {PREPROCESSED_DATA_PATH}...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    df["similar"] = df["similar"].apply(json.loads)
    full_ds = Dataset.from_pandas(df)

    split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds['train'].shuffle(seed=42).select(range(min(5000, len(split_ds['train']))))
    eval_ds = split_ds['test'].shuffle(seed=42).select(range(min(500, len(split_ds['test']))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def process_and_label_function(examples):
        tokenized_inputs = tokenizer(examples["post"], truncation=True, padding=False, max_length=512)
        labels = []
        for i, similar_phrases_dict in enumerate(examples["similar"]):
            essential_words = set(word for phrase in similar_phrases_dict.keys() for word in phrase.lower().split())
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            token_labels = []
            for word_idx in word_ids:
                if word_idx is None:
                    token_labels.append(-100)
                elif word_idx != previous_word_idx:
                    word = tokenized_inputs.words(batch_index=i)[word_idx]
                    if word and word.lower() in essential_words:
                        token_labels.append(LABELS.index("ESSENTIAL"))
                    else:
                        token_labels.append(LABELS.index("FLUFF"))
                else:
                    token_labels.append(-100)
                previous_word_idx = word_idx
            labels.append(token_labels)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing and labeling datasets for training...")
    original_columns = train_ds.column_names
    tokenized_train_ds = train_ds.map(process_and_label_function, batched=True, remove_columns=original_columns)
    tokenized_eval_ds = eval_ds.map(process_and_label_function, batched=True, remove_columns=original_columns)

    return tokenized_train_ds, tokenized_eval_ds, tokenizer


if __name__ == '__main__':
    create_preprocessed_csv()