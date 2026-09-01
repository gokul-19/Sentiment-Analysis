"""Training and inference utilities for the sentiment dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "data" / "sentiment_reviews.csv"
DEFAULT_MODEL = ROOT / "models" / "sentiment_pipeline.joblib"


def clean_text(value: Any) -> str:
    """Apply lightweight, reproducible preprocessing without external downloads."""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    data["text"] = data["text"].map(clean_text)
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if not set(data["label"]).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 (negative) or 1 (positive).")
    return data


def build_pipeline(model_name: str = "logistic_regression") -> Pipeline:
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced") if model_name == "logistic_regression" else MultinomialNB()
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("classifier", classifier),
    ])


def train_and_compare(data: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=test_size, random_state=random_state, stratify=data["label"]
    )
    candidates = {"Logistic Regression": "logistic_regression", "Multinomial Naive Bayes": "naive_bayes"}
    rows, trained = [], {}
    for display_name, model_name in candidates.items():
        model = build_pipeline(model_name)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rows.append({"Model": display_name, "Accuracy": accuracy_score(y_test, predicted), "F1 score": f1_score(y_test, predicted)})
        trained[display_name] = (model, predicted)
    results = pd.DataFrame(rows).sort_values(["F1 score", "Accuracy"], ascending=False).reset_index(drop=True)
    winner = results.iloc[0]["Model"]
    model, predicted = trained[winner]
    metadata = {
        "best_model": winner,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_score": float(f1_score(y_test, predicted)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(y_test, predicted, target_names=["Negative", "Positive"], output_dict=True),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }
    return model, results, metadata


def save_model(model: Pipeline, path: str | Path = DEFAULT_MODEL) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_or_train_model(dataset_path: str | Path = DEFAULT_DATASET, model_path: str | Path = DEFAULT_MODEL) -> Pipeline:
    target = Path(model_path)
    if target.exists():
        return joblib.load(target)
    model, _, _ = train_and_compare(load_dataset(dataset_path))
    save_model(model, target)
    return model


def predict_sentiment(model: Pipeline, text: str) -> tuple[str, float]:
    if not clean_text(text):
        raise ValueError("Enter some review text before analysis.")
    probability = float(model.predict_proba([clean_text(text)])[0][1])
    return ("Positive" if probability >= 0.5 else "Negative"), max(probability, 1 - probability)
