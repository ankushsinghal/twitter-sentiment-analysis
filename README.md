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

## Experiment 02: BERTweet fine-tuning

This trains `vinai/bertweet-base`, a tweet-specialized BERT-base model, on the
TweetEval training split. The best checkpoint is selected by **validation
macro-F1**; the test split is never loaded for this experiment.

In the same Colab runtime, run:

```bash
!pip install -q -r requirements.txt
!python src/train_bertweet.py
```

The Hugging Face unauthenticated-download warning is optional: downloading this
public model works without a token. You may authenticate with an HF token to
avoid rate limits when repeating runs.

The default configuration is four epochs, learning rate `2e-5`, batch sizes 16
and 32, `max_length=128`, linear warmup/decay, and fp16 automatically enabled
when Colab provides a CUDA GPU. Output is written to:

```text
artifacts/bertweet/best_model/
artifacts/bertweet/validation_report.json
```

If a T4 runs out of memory, rerun with a smaller training batch size:

```bash
!python src/train_bertweet.py --train-batch-size 8 --eval-batch-size 16
```

Do not run on the official test split until choosing this configuration and any
hyperparameter variants using validation macro-F1.
