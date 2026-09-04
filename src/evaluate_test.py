"""Evaluate one selected BERTweet checkpoint on a TweetEval split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATASET_ID = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing the selected saved model (the best_model directory).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Evaluation split. Use validation to verify a saved checkpoint; use test once for final reporting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bertweet/test_report.json"),
        help="Path for the test-set JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Saved model directory not found: {args.model_dir}")

    raw_dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    evaluation_dataset = raw_dataset[args.split]
    label_names = list(raw_dataset["train"].features["label"].names)
    labels = list(range(len(label_names)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    def tokenize(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_evaluation = evaluation_dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    evaluation_args = TrainingArguments(
        output_dir=str(args.output.parent / ".evaluation_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    evaluator = Trainer(
        model=model,
        args=evaluation_args,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    output = evaluator.predict(tokenized_evaluation)
    references = output.label_ids
    predictions = np.argmax(output.predictions, axis=-1)
    report = {
        "experiment": f"bertweet_{args.split}_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "configuration": DATASET_CONFIG,
        "evaluated_split": args.split,
        "model_dir": str(args.model_dir),
        "inference_device": str(evaluator.args.device),
        "evaluation_examples": len(references),
        "metrics": {
            "accuracy": accuracy_score(references, predictions),
            "macro_f1": f1_score(references, predictions, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(references, predictions, labels=labels, average="weighted", zero_division=0),
            "per_class": classification_report(
                references,
                predictions,
                labels=labels,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(references, predictions, labels=labels).tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(f"{args.split.title()} accuracy:    {metrics['accuracy']:.4f}")
    print(f"{args.split.title()} macro F1:    {metrics['macro_f1']:.4f}")
    print(f"{args.split.title()} weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
