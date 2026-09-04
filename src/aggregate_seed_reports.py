"""Aggregate validation reports from repeated BERTweet training runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to BERTweet validation_report.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bertweet/seed_summary.json"),
        help="Path for the aggregate JSON report.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    required_fields = ("experiment", "hyperparameters", "metrics")
    if any(field not in report for field in required_fields) or report["experiment"] != "bertweet_finetuning":
        raise ValueError(f"Not a BERTweet validation report: {path}")
    return report


def main() -> None:
    args = parse_args()
    reports = [load_report(path) for path in args.reports]
    seeds = [report["hyperparameters"]["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each report must come from a different random seed.")

    metric_names = ("accuracy", "macro_f1", "weighted_f1")
    summary_metrics = {
        metric: {
            "mean": mean(report["metrics"][metric] for report in reports),
            "sample_standard_deviation": (
                stdev(report["metrics"][metric] for report in reports) if len(reports) > 1 else 0.0
            ),
        }
        for metric in metric_names
    }
    summary = {
        "experiment": "bertweet_repeated_seeds",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "seed": report["hyperparameters"]["seed"],
                "model_id": report["model_id"],
                "validation_metrics": report["metrics"],
            }
            for report in reports
        ],
        "aggregate_validation_metrics": summary_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Seeds: {', '.join(map(str, seeds))}")
    for metric, values in summary_metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['sample_standard_deviation']:.4f}")
    print(f"Saved summary: {args.output}")


if __name__ == "__main__":
    main()
