# Healthcare Text Classification with BioClinicalBERT

This repository provides a reproducible research template for classifying short healthcare narratives with [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT). It includes synthetic, non-sensitive demonstration data, training, evaluation, prediction, and unit tests.

## Task

The included demonstration classifies text into four educational categories:

- `cardiology`
- `neurology`
- `respiratory`
- `gastroenterology`

These labels and examples are synthetic. Replace them with an appropriate, authorized dataset for real research.

## Structure

```text
healthcare-text-classification-bert/
├── data/synthetic_clinical_notes.csv
├── src/healthcare_nlp/{data.py,model.py,utils.py}
├── tests/test_data.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py --data data/synthetic_clinical_notes.csv --output outputs/model --epochs 3
```

The script creates a stratified train/test split, fine-tunes BioClinicalBERT, and saves accuracy, macro precision, recall, F1, a classification report, and a confusion matrix under `outputs/model/`.

## Predict

```bash
python predict.py --model outputs/model --text "The patient reports persistent cough and shortness of breath."
```

## Use another dataset

Supply a UTF-8 CSV containing `text` and `label` columns:

```bash
python train.py --data path/to/data.csv --text-column text --label-column label
```

Do not commit protected health information, credentials, or restricted datasets to GitHub.

## Limitations and responsible use

The bundled dataset is intentionally tiny and supports pipeline testing—not meaningful clinical performance. Results should not be interpreted as medical evidence. This software is for academic and educational use only and is not a diagnostic device.

## Author

**Sara Yavari** — research interests include healthcare AI, medical image and text analysis, NLP, computer vision, and deep learning.

## License

MIT License. See `LICENSE`.
