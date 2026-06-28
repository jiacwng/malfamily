"""Random Forest malware-type classifier.

One model per architecture (the trainer fits x86 and arm64 separately, since
their feature vectors differ in width), with evaluation report.

Public API:
    train(X, y)        -> (model, EvalReport)
    predict(model, v)  -> (label, confidence, margin, runner_up, runner_up_confidence)
    save(model, path) / load(path)
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class EvalReport:
    """Held-out evaluation w/ accuracy,precision,recall,f1 and confusion matrix."""

    accuracy: float
    baseline_accuracy: float  # The baseline accuracy that our model needs to beat.
    macro_f1: float
    per_family: dict[str, dict[str, float]]
    confusion: np.ndarray  # Confusion matrix (rows = actual class, columns = predicted class)
    labels: list[str]
    margins: np.ndarray
    correct: np.ndarray


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    n_estimators: int = 300,
    seed: int = 0,
) -> tuple[RandomForestClassifier, EvalReport]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, _evaluate(model, X_test, y_test)


def _evaluate(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> EvalReport:
    y_pred = model.predict(X_test)
    labels = sorted(set(y_test.tolist()))
    # function from sklearn for evaluation
    rep = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_family = {
        fam: {
            "precision": float(rep[fam]["precision"]),
            "recall": float(rep[fam]["recall"]),
            "f1": float(rep[fam]["f1-score"]),
            "support": float(rep[fam]["support"]),
        }
        for fam in labels
    }
    # dummy that always predict the most common family
    counts = Counter(y_test.tolist())
    baseline = counts.most_common(1)[0][1] / len(y_test)

    # decision margin (p1 - p2) + correctness, for confidence calibration
    proba = model.predict_proba(X_test)
    top2 = np.sort(proba, axis=1)[:, -2:]  # columns are (p2, p1)
    margins = (top2[:, 1] - top2[:, 0]).astype(float)
    correct = y_pred == y_test

    return EvalReport(
        accuracy=float(rep["accuracy"]),
        baseline_accuracy=float(baseline),
        macro_f1=float(rep["macro avg"]["f1-score"]),
        per_family=per_family,
        confusion=confusion_matrix(y_test, y_pred, labels=labels),
        labels=labels,
        margins=margins,
        correct=correct,
    )


def predict(
    model: RandomForestClassifier, vector: np.ndarray
) -> tuple[str, float, float, str, float]:
    """Top type + confidence (p1), margin (p1-p2), and the runner-up type + its p2.

    The runner-up is returned so callers can show how close the second guess was. The
    margin (p1-p2) is the confidence signal, used instead of a flat 50% threshold because
    votes get split across many classes.
    """
    # scikit-learn models require a 2d table
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]  # classes from most to least likely
    i = int(order[0])
    p1 = float(proba[i])
    if proba.size > 1:
        j = int(order[1])
        p2 = float(proba[j])
        return str(model.classes_[i]), p1, p1 - p2, str(model.classes_[j]), p2
    return str(model.classes_[i]), p1, p1, "", 0.0


def save(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load(path: Path) -> RandomForestClassifier:
    with path.open("rb") as fh:
        return pickle.load(fh)
