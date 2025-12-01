# src/nlp_eval.py
import pandas as pd
import numpy as np
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import joblib
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/simulated/journals.csv"
MODEL_DIR = "src/models/nlp_model"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # same label ordering as training
    labels = sorted(df['emotion'].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df['label'] = df['emotion'].map(label2id)
    return df, labels

def main():
    df, labels = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, function_to_apply="softmax")
    # sample test set (or full)
    ds = Dataset.from_pandas(df[['journal','label']])
    # simple 80/20 split reproducible
    test = ds.select(range(int(len(ds)*0.85), len(ds)))
    y_true = []
    y_pred = []
    for i in range(len(test)):
        row = test[i]
        pred = pipe(row['journal'])[0]
        # pipeline returns label like "LABEL_2" or the string label; check
        lab = pred['label']
        # handle both cases
        if lab.startswith("LABEL_"):
            pred_idx = int(lab.split("_")[-1])
        else:
            # try to map string label
            try:
                pred_idx = labels.index(lab)
            except:
                pred_idx = int(lab)
        y_true.append(row['label'])
        y_pred.append(pred_idx)

    print(classification_report(y_true, y_pred, target_names=labels))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
