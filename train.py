import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from healthcare_nlp.data import ClinicalTextDataset, load_labeled_csv
from healthcare_nlp.utils import save_json, set_seed


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return np.asarray(targets), np.asarray(predictions)


def main(args):
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_labeled_csv(args.data, args.text_column, args.label_column)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame[args.label_column])
    if len(encoder.classes_) < 2:
        raise ValueError("At least two label classes are required.")
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data = ClinicalTextDataset(frame.iloc[train_idx][args.text_column], labels[train_idx], tokenizer, args.max_length)
    test_data = ClinicalTextDataset(frame.iloc[test_idx][args.text_column], labels[test_idx], tokenizer, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {running_loss / max(len(train_loader), 1):.4f}")
    y_true, y_pred = evaluate(model, test_loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}
    save_json(metrics, output / "metrics.json")
    save_json(classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0), output / "classification_report.json")
    with (output / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(encoder.classes_.tolist(), handle, indent=2)
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=200); plt.close()
    model.save_pretrained(output); tokenizer.save_pretrained(output)
    print(json.dumps(metrics, indent=2)); print(f"Saved model and results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BioClinicalBERT for labeled healthcare text classification.")
    parser.add_argument("--data", default="data/synthetic_clinical_notes.csv")
    parser.add_argument("--output", default="outputs/model")
    parser.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
