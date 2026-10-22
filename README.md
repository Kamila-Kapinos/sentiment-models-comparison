# Sentiment Analysis of Amazon Reviews: SVM vs BiLSTM vs RoBERTa

Code for my master's thesis, which compares three generations of NLP methods on the same task: classifying consumer reviews as **negative**, **neutral** or **positive**.

| Approach | Model | Text representation |
|---|---|---|
| Classical ML | Linear SVM (`LinearSVC`) | TF-IDF, word n-grams (1–3) |
| Recurrent neural network | 2-layer BiLSTM (PyTorch) | Frozen GloVe 6B 100d embeddings |
| Transformer | `roberta-base`, fine-tuned (Hugging Face) | Subword tokens, contextual embeddings |

All three models are trained and evaluated on the same data, with the same stratified 20% hold-out test set (`random_state=42`), so their results can be compared directly.

## Results

Test set: 17,903 reviews (balanced, about 5,970 per class).

| Model | Accuracy | F1-macro | F1 Negative | F1 Neutral | F1 Positive | Training time |
|---|---:|---:|---:|---:|---:|---|
| SVM + TF-IDF | 0.78 | 0.7768 | 0.78 | 0.70 | 0.86 | ~8 min, grid search on CPU (90 fits) |
| BiLSTM + GloVe | 0.78 | 0.7841 | 0.78 | 0.71 | 0.87 | ~31 min (Apple MPS, 19 epochs) |
| **RoBERTa-base** | **0.83** | **0.8331** | **0.82** | **0.76** | **0.91** | ~51 min (Colab T4 GPU, 4 epochs) |

Key findings:
- **Accuracy goes up as the model sees more context.** RoBERTa scored about 5.5 F1-macro points higher than SVM and about 5 points higher than BiLSTM.
- **The neutral class (3★) was the hardest for every model.** Neutral reviews often mix praise and complaints, so most errors are confusions with the neighbouring class. Strongly negative and strongly positive reviews were identified much more reliably.
- **Better accuracy costs more compute.** The SVM trains in minutes on a laptop CPU. RoBERTa needs a GPU and is far more expensive to train and run, which matters when choosing a model for production.

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (568,454 reviews). Preprocessing is in [`clean_dataset.ipynb`](clean_dataset.ipynb):

1. `Summary` and `Text` are joined into one `full_text` field.
2. Star ratings are mapped to sentiment: 1–2 → Negative, 3 → Neutral, 4–5 → Positive.
3. Duplicate and empty reviews are removed.
4. Each class is undersampled to the size of the smallest one (neutral), giving **89,511 reviews (29,837 per class)**.

The processed file `data/processed_reviews.csv` is included in the repo. To rerun preprocessing, put the raw Kaggle `Reviews.csv` at `data/amazon_reviews.csv`.

> Because the classes are balanced, the metrics describe performance on a balanced distribution, not on the natural, heavily positive distribution of Amazon ratings.

## Repository structure

| Notebook | Contents |
|---|---|
| [`clean_dataset.ipynb`](clean_dataset.ipynb) | Data cleaning, label mapping, class balancing |
| [`svm_sentiment_final.ipynb`](svm_sentiment_final.ipynb) | TF-IDF + LinearSVC, 5-fold CV baseline, `GridSearchCV` over `C`, vocabulary size and n-gram range, error analysis |
| [`bilstm_glove_sentiment_final.ipynb`](bilstm_glove_sentiment_final.ipynb) | BiLSTM with masked mean pooling, early stopping, `ReduceLROnPlateau`, learning curves, error analysis |
| [`roberta_base_final.ipynb`](roberta_base_final.ipynb) | `roberta-base` fine-tuning with AMP, gradient accumulation, linear warmup, checkpointing and resume support, error analysis, final comparison |

Data splits:
- **SVM:** 80% train (hyperparameters picked with 5-fold CV) and 20% test.
- **BiLSTM and RoBERTa:** 70% train, 10% validation (for early stopping) and 20% test. The test set is the same 20% that the SVM uses.

The notebooks' comments and markdown are in Polish.

## Setup

```bash
git clone git@github.com:Kamila-Kapinos/sentiment-models-comparison.git
cd sentiment-models-comparison

python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name=sent-mag-env --display-name "Python (Sent-Mag)"
```

Extra files:
- **BiLSTM:** download [GloVe 6B](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.100d.txt` in `data/`.
- **RoBERTa:** the notebook was run in Google Colab with a T4 GPU. It reads `sample_data/processed_reviews.csv`. To run it locally, change `DATA_PATH` to `data/processed_reviews.csv`. Model checkpoints are saved to `models/`, which is not tracked in git.

Tested with Python 3.10.

## Tech stack

Python · scikit-learn · PyTorch · Hugging Face Transformers · pandas · NumPy · matplotlib · seaborn · Jupyter

## License

[MIT](LICENSE)
