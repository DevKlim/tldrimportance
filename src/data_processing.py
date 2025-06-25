from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# We'll use a smaller, faster model for the real-time app
MODEL_CHECKPOINT = "distilbert-base-uncased"
# The labels we want to predict
LABELS = ["FLUFF", "ESSENTIAL"] # 0: FLUFF, 1: ESSENTIAL

def create_and_process_dataset():
    """
    Loads the TLDR dataset and processes it for token classification.
    """
    # 1. Load the dataset
    print("Loading tldr dataset...")
    ds = load_dataset("trl-lib/tldr")
    # For demonstration, let's work with a smaller subset
    train_ds = ds["train"].shuffle(seed=42).select(range(5000))
    eval_ds = ds["validation"].shuffle(seed=42).select(range(500))
    
    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def process_and_label_function(examples):
        """
        This is the core function to create labels.
        It tokenizes the prompt and assigns a label (0 or 1) to each token.
        """
        # Tokenize the long posts (prompts)
        tokenized_inputs = tokenizer(
            examples["prompt"], 
            truncation=True, 
            padding=False, # We'll pad later dynamically
            max_length=512,
            # We need word IDs to map labels to tokens
            return_word_ids=True
        )

        labels = []
        for i, prompt in enumerate(examples["prompt"]):
            # Get the TLDR summary for this prompt
            tldr_summary = examples["label"][i]
            # Create a set of unique words from the TLDR for fast lookup
            # We lowercase and split by space for a simple heuristic
            tldr_words = set(tldr_summary.lower().split())

            # Get the word IDs for the current tokenized prompt
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            token_labels = []

            for word_idx in word_ids:
                # Special tokens like [CLS], [SEP] get a -100 label, which is ignored by the loss function
                if word_idx is None:
                    token_labels.append(-100)
                # We only label the first token of a given word
                elif word_idx != previous_word_idx:
                    # Get the full word from the original prompt using the word_idx
                    word = tokenized_inputs.words(batch_index=i)[word_idx]
                    # Check if the word (or its lowercased version) is in the TLDR
                    if word.lower() in tldr_words:
                        token_labels.append(1)  # ESSENTIAL
                    else:
                        token_labels.append(0)  # FLUFF
                # Subsequent tokens of the same word also get -100
                else:
                    token_labels.append(-100)
                previous_word_idx = word_idx
            
            labels.append(token_labels)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Processing and labeling datasets...")
    tokenized_train_ds = train_ds.map(process_and_label_function, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval_ds = eval_ds.map(process_and_label_function, batched=True, remove_columns=eval_ds.column_names)

    return tokenized_train_ds, tokenized_eval_ds, tokenizer

if __name__ == '__main__':
    # You can run this file directly to test the data processing
    train_data, eval_data, tokenizer = create_and_process_dataset()
    print("\nSample processed data:")
    sample = train_data[0]
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(sample['input_ids'])}")
    print(f"Labels: {sample['labels']}")