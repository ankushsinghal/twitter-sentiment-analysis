# Twitter Sentiment Analysis

This project fine-tunes a Twitter-specialized BERT model for three-way English
tweet sentiment: negative, neutral, and positive.

## Experiment 01: Majority-class baseline

The first experiment does **not** train a model. It predicts the label most
common in TweetEval's training split for every tweet in the validation split.
It provides a reproducible lower bound and confirms that the data, labels, and
macro-F1 evaluation pipeline work before a BERTweet run.

The official test split must remain untouched until model selection is complete.

### Run in Google Colab

1. Create a new Colab notebook and select **Runtime → Change runtime type → T4 GPU**.
2. Upload this project or clone its Git repository in Colab.
3. From the project directory, run:

```bash
!pip install -q -r requirements.txt
!python src/run_majority_baseline.py
```

The run downloads `cardiffnlp/tweet_eval` with the `sentiment` configuration,
evaluates on `validation`, and writes a JSON report to
`artifacts/reports/majority_baseline_validation.json`.

Use `--split test` only once, after the final BERTweet configuration has been
selected on validation data.

```bash
!python src/run_majority_baseline.py --split test
```

### Success criteria

- The script runs without manually downloading data.
- The report records the label distribution and accuracy, macro-F1, weighted-F1,
  per-class metrics, and confusion matrix.
- Future experiments use the same validation split and select models by
  validation macro-F1.

## Next experiment

Fine-tune `vinai/bertweet-base` on TweetEval sentiment using the exact same
split and macro-F1 model-selection metric.
