"""Fine-tune BERTweet on TweetEval sentiment and evaluate on validation data."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "vinai/bertweet-base"
DATASET_ID = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bertweet"))
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Use fp16. Defaults to enabled when CUDA is available (for example, Colab T4).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_factory(label_names: list[str]):
    labels = list(range(len(label_names)))

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, references = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(references, predictions),
            "macro_f1": f1_score(references, predictions, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(references, predictions, labels=labels, average="weighted", zero_division=0),
        }

    return compute_metrics


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    def tokenize(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def validation_report(
    predictions: np.ndarray, references: np.ndarray, label_names: list[str], args: argparse.Namespace, trainer: Trainer
) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    predicted_labels = np.argmax(predictions, axis=-1)
    return {
        "experiment": "bertweet_finetuning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "configuration": DATASET_CONFIG,
        "evaluated_split": "validation",
        "model_id": args.model_id,
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "fp16": args.use_fp16,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "metrics": {
            "accuracy": accuracy_score(references, predicted_labels),
            "macro_f1": f1_score(references, predicted_labels, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(references, predicted_labels, labels=labels, average="weighted", zero_division=0),
            "per_class": classification_report(
                references,
                predicted_labels,
                labels=labels,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(references, predicted_labels, labels=labels).tolist(),
        },
    }


def make_training_arguments(args: argparse.Namespace, train_examples: int) -> TrainingArguments:
    """Build TrainingArguments across compatible Transformers releases.

    Some Colab images expose ``warmup_steps`` but not ``warmup_ratio``. The
    equivalent number of warmup steps is used in that case.
    """
    argument_names = inspect.signature(TrainingArguments).parameters
    steps_per_epoch = math.ceil(train_examples / args.train_batch_size)
    warmup_steps = math.ceil(steps_per_epoch * args.epochs * args.warmup_ratio)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": args.use_fp16,
        "report_to": "none",
        "seed": args.seed,
    }
    training_kwargs["eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"] = "epoch"
    if "warmup_ratio" in argument_names:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        training_kwargs["warmup_steps"] = warmup_steps
    if "data_seed" in argument_names:
        training_kwargs["data_seed"] = args.seed
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.use_fp16 and not torch.cuda.is_available():
        raise RuntimeError("--use-fp16 requires a CUDA GPU. In Colab, select a T4 GPU runtime.")

    raw_dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    label_names = list(raw_dataset["train"].features["label"].names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, normalization=True)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, args.max_length)

    label_to_id = {name: index for index, name in enumerate(label_names)}
    id_to_label = {index: name for name, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(label_names),
        label2id=label_to_id,
        id2label=id_to_label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = make_training_arguments(args, len(tokenized_dataset["train"]))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_factory(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()

    model_dir = args.output_dir / "best_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    output = trainer.predict(tokenized_dataset["validation"])
    report = validation_report(output.predictions, output.label_ids, label_names, args, trainer)
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(f"Validation accuracy:    {metrics['accuracy']:.4f}")
    print(f"Validation macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Validation weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved model: {model_dir}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
