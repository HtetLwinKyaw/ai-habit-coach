# src/nlp_train.py
import pandas as pd
import os
import numpy as np
import torch
from datasets import Dataset, ClassLabel


from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

DATA_PATH = "data/simulated/journals.csv"
MODEL_DIR = "src/models/nlp_model"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # ensure consistent ordering of labels
    labels = sorted(df['emotion'].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    df = df[['journal','emotion']].copy()
    df['label'] = df['emotion'].map(label2id)
    return df, labels

def preprocess(df, tokenizer):
    # convert to HF Dataset
    ds = Dataset.from_pandas(df[['journal','label']], preserve_index=False)

    # Convert label column to ClassLabel so datasets.train_test_split can stratify
    labels = sorted(df['label'].unique())
    class_label = ClassLabel(num_classes=int(max(labels)+1))  # assumes labels are 0..num-1
    ds = ds.cast_column("label", class_label)

    # tokenize
    def tok(batch):
        return tokenizer(batch['journal'], truncation=True, padding='max_length', max_length=128)
    ds = ds.map(tok, batched=True)

    # split to train/test (stratify by label)
    ds = ds.train_test_split(test_size=0.15, stratify_by_column='label', seed=42)
    return ds

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

def main():
    print("Loading data...")
    df, labels = load_data()
    num_labels = len(labels)
    print(f"Found {len(df)} rows, labels: {labels}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("Preprocessing dataset...")
    ds = preprocess(df, tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    # small, CPU-friendly args by default — change per_device_train_batch_size if you have a GPU
    training_args = TrainingArguments(
        output_dir="./tmp_nlp",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        logging_steps=20,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    # save labels
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(labels))
    print("Saved model to", MODEL_DIR)

if __name__ == "__main__":
    main()
