import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.preprocess_data import get_tokenized_datasets, MODEL_CHECKPOINT, LABELS

def compute_metrics(p):
    """
    Computes precision, recall, F1, and accuracy for the token classification task.
    It ignores the -100 label for pads/sub-word tokens.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Flatten the predictions and labels, ignoring the -100 index
    true_predictions = [
        p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]

    # Calculate metrics for the "ESSENTIAL" class (label=1)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="binary", pos_label=LABELS.index("ESSENTIAL"))
    acc = accuracy_score(true_labels, true_predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def train_model():
    """
    Main function to orchestrate the training process.
    """
    # 1. Load and process data using the consolidated script
    print("Loading and tokenizing datasets...")
    train_dataset, eval_dataset, tokenizer = get_tokenized_datasets()

    # 2. Define the model
    # We create a mapping from label name to ID and vice-versa for clarity
    id2label = {i: label for i, label in enumerate(LABELS)}
    label2id = {label: i for i, label in enumerate(LABELS)}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # 3. Define Training Arguments
    # The error was caused by an old `transformers` version. With the pinned
    # version, these modern arguments will work correctly.
    output_dir = "/app/results"
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    # 4. Data Collator
    # This will dynamically pad the inputs and labels in each batch.
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting model training...")
    trainer.train()

    # 6. Save the final model and tokenizer
    print("Training finished. Saving best model...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model()