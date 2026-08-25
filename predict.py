import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(model_path: str, text: str):
    path = Path(model_path)
    with (path / "labels.json").open(encoding="utf-8") as handle:
        labels = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    encoded = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    print(json.dumps({"label": labels[index], "confidence": round(float(probabilities[index]), 4)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/model")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    main(args.model, args.text)
