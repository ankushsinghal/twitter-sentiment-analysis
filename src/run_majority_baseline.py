"""Evaluate a majority-class baseline on TweetEval sentiment.

This intentionally simple experiment verifies dataset access, label handling, and
metrics before any transformer model is trained.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Split to evaluate. Keep the default validation split while developing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory in which to write the JSON report.",
    )
    return parser.parse_args()


def build_report(train_labels: list[int], eval_labels: list[int], label_names: list[str]) -> dict[str, Any]:
    """Return metrics for predicting the training-set majority class everywhere."""
    counts = Counter(train_labels)
    majority_label, majority_count = counts.most_common(1)[0]
    predictions = [majority_label] * len(eval_labels)
    labels = list(range(len(label_names)))

    return {
        "experiment": "majority_class_baseline",
        "dataset": "cardiffnlp/tweet_eval",
        "configuration": "sentiment",
        "prediction_rule": f"Predict '{label_names[majority_label]}' for every tweet.",
        "majority_label_id": majority_label,
        "majority_label": label_names[majority_label],
        "train_examples": len(train_labels),
        "evaluation_examples": len(eval_labels),
        "train_class_counts": {label_names[label]: counts.get(label, 0) for label in labels},
        "metrics": {
            "accuracy": accuracy_score(eval_labels, predictions),
            "macro_f1": f1_score(eval_labels, predictions, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(eval_labels, predictions, labels=labels, average="weighted", zero_division=0),
            "per_class": classification_report(
                eval_labels,
                predictions,
                labels=labels,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(eval_labels, predictions, labels=labels).tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    label_names = list(dataset["train"].features["label"].names)
    report = build_report(
        train_labels=list(dataset["train"]["label"]),
        eval_labels=list(dataset[args.split]["label"]),
        label_names=label_names,
    )
    report["evaluated_split"] = args.split
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"majority_baseline_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(f"Majority label: {report['majority_label']} ({report['majority_label_id']})")
    print(f"Validation split: {args.split} ({report['evaluation_examples']} tweets)")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
