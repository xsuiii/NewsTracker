"""
Fake-News-Klassifikation mit BERT und dem LIAR-Datensatz.

Dieses Skript trainiert (fine-tuned) ein vortrainiertes BERT-Modell von
HuggingFace ("bert-base-uncased") auf dem LIAR-Datensatz, um Nachrichten-
Aussagen binär als "Fake" oder "Echt" zu klassifizieren.

Der LIAR-Datensatz enthält ursprünglich 6 Wahrheitsstufen
(pants-fire, false, barely-true, half-true, mostly-true, true).
Für die binäre Klassifikation werden diese auf zwei Klassen abgebildet:
    - FAKE  (0): pants-fire, false, barely-true
    - ECHT  (1): half-true, mostly-true, true

Verwendung:
    # Modell trainieren und auswerten
    python backend/fake_news_detection.py --mode train

    # Einzelne Vorhersage treffen (nach dem Training)
    python backend/fake_news_detection.py --mode predict \
        --text "Die Erde ist eine Scheibe."
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
MODEL_NAME = "bert-base-uncased"           # Vortrainiertes BERT-Modell
OUTPUT_DIR = "models/bert_fake_news"       # Pfad für gespeichertes Modell
MAX_LENGTH = 128                           # Maximale Tokenlänge pro Eingabe
NUM_LABELS = 2                             # Binäre Klassifikation: Fake/Echt

# Mapping der 6 LIAR-Labels auf binäre Klassen
# Original-Label-IDs im LIAR-Datensatz:
#   0 = false, 1 = half-true, 2 = mostly-true, 3 = true,
#   4 = barely-true, 5 = pants-fire
LIAR_TO_BINARY: Dict[int, int] = {
    0: 0,  # false        -> FAKE
    4: 0,  # barely-true  -> FAKE
    5: 0,  # pants-fire   -> FAKE
    1: 1,  # half-true    -> ECHT
    2: 1,  # mostly-true  -> ECHT
    3: 1,  # true         -> ECHT
}

LABEL_NAMES: Dict[int, str] = {0: "Fake", 1: "Echt"}


# --------------------------------------------------------------------------- #
# Daten laden und vorbereiten
# --------------------------------------------------------------------------- #
def load_and_prepare_dataset(tokenizer):
    """Lädt den LIAR-Datensatz und tokenisiert ihn für BERT.

    Der LIAR-Datensatz wird direkt über die HuggingFace `datasets`-Bibliothek
    bezogen und in Train/Validation/Test-Splits aufgeteilt.
    """
    # LIAR-Datensatz von HuggingFace Hub laden (enthält train/validation/test)
    dataset = load_dataset("liar")

    def map_labels(example):
        # 6-Klassen-Label auf binäres Label (0 = Fake, 1 = Echt) abbilden
        example["label"] = LIAR_TO_BINARY[example["label"]]
        return example

    def tokenize(batch):
        # Die "statement"-Spalte enthält die zu klassifizierende Aussage
        return tokenizer(
            batch["statement"],
            truncation=True,
            padding=False,             # Padding übernimmt der DataCollator
            max_length=MAX_LENGTH,
        )

    # Labels umbiegen und tokenisieren
    dataset = dataset.map(map_labels)
    dataset = dataset.map(tokenize, batched=True)

    # Nur die für das Training nötigen Spalten behalten
    keep_cols = ["input_ids", "attention_mask", "label"]
    dataset = dataset.remove_columns(
        [c for c in dataset["train"].column_names if c not in keep_cols]
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    return dataset


# --------------------------------------------------------------------------- #
# Evaluations-Metriken
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    """Berechnet Accuracy, Precision, Recall und F1-Score."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train():
    """Trainiert BERT auf dem LIAR-Datensatz und speichert das Modell."""
    # Tokenizer und Modell mit binärem Klassifikationskopf laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    # Daten laden und tokenisieren
    dataset = load_and_prepare_dataset(tokenizer)

    # Dynamisches Padding pro Batch (effizienter als statisches Padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainingsparameter
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),   # Mixed Precision nur mit GPU
    )

    # Trainer instanziieren und Training starten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Auswertung auf dem Test-Split
    test_metrics = trainer.evaluate(dataset["test"])
    print("Testergebnisse:", test_metrics)

    # Bestes Modell + Tokenizer dauerhaft speichern
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modell gespeichert unter: {OUTPUT_DIR}")


# --------------------------------------------------------------------------- #
# Inferenz / Vorhersage
# --------------------------------------------------------------------------- #
def predict(texts: List[str], model_dir: str = OUTPUT_DIR) -> List[Dict[str, float]]:
    """Klassifiziert eine Liste von Texten als 'Fake' oder 'Echt'.

    Lädt das fine-getunte Modell aus `model_dir` (Fallback: vortrainiertes
    BERT, falls noch kein Training stattgefunden hat) und gibt Label samt
    Konfidenz zurück.
    """
    source = model_dir if os.path.isdir(model_dir) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Eingaben tokenisieren und auf das Zielgerät verschieben
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(device)

    # Vorhersage ohne Gradientenberechnung (spart Speicher)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)

    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append({
            "text": text,
            "label": LABEL_NAMES[int(pred)],
            "confidence": float(prob[int(pred)]),
        })
    return results


def detect_fake_news(article: str) -> str:
    """Komfortfunktion: Klassifiziert einen einzelnen Artikel.

    Wird vom restlichen Backend (z. B. main.py) für eine schnelle Prüfung
    eines einzelnen Textes verwendet.
    """
    result = predict([article])[0]
    return f"{result['label']} (Konfidenz: {result['confidence']:.2%})"


# --------------------------------------------------------------------------- #
# CLI-Einstiegspunkt
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="BERT-basierte Fake-News-Klassifikation mit LIAR-Datensatz."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: Modell trainieren. predict: Vorhersage für --text treffen.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Zu klassifizierender Text (nur im Modus 'predict').",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        if not args.text:
            parser.error("--text wird im Modus 'predict' benötigt.")
        print(detect_fake_news(args.text))


if __name__ == "__main__":
    main()
