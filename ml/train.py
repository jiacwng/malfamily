"""Train + evaluate the per-architecture family classifiers.

Usage:
    python -m ml.train [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ml import classifier
from ml.classifier import EvalReport
from ml.dataset import Dataset, build_dataset

_MODEL_DIR = Path(__file__).resolve().parent  # ml/


def _model_path(arch: str) -> Path:
    return _MODEL_DIR / f"model_{arch}.pkl"


def _trainable(ds: Dataset) -> bool:
    """Need >=2 families to classify between; the stratified-split edge cases (a
    family too small to split) are caught by the try/except around train()."""
    return len(set(ds.y.tolist())) >= 2


def _print_report(arch: str, ds: Dataset, report: EvalReport) -> None:
    print(f"\n=== {arch}: {ds.X.shape[0]} samples x {ds.X.shape[1]} features ===")
    for fam, n in sorted(Counter(ds.y.tolist()).items()):
        print(f"    {fam:24s} {n}")
    print(f"  baseline (majority class) : {report.baseline_accuracy:6.1%}")
    print(f"  random forest accuracy    : {report.accuracy:6.1%}")
    print(f"  macro F1                  : {report.macro_f1:6.3f}")
    print("  per-family  precision / recall / f1:")
    for fam in report.labels:
        s = report.per_family[fam]
        print(f"    {fam:24s} {s['precision']:.2f} / {s['recall']:.2f} / {s['f1']:.2f}")
    print("  confusion (rows=true, cols=pred):")
    print("    " + " " * 16 + "".join(f"{lab[:8]:>9s}" for lab in report.labels))
    for i, lab in enumerate(report.labels):
        cells = "".join(f"{int(report.confusion[i, j]):>9d}" for j in range(len(report.labels)))
        print(f"    {lab[:16]:16s}{cells}")
    # calibration: predict-time low-confidence cutoff comes from where accuracy
    # falls off as the decision margin (p1-p2) shrinks.
    print("  margin calibration (accuracy by decision margin p1-p2):")
    for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 1.01)):
        mask = (report.margins >= lo) & (report.margins < hi)
        n = int(mask.sum())
        acc = float(report.correct[mask].mean()) if n else 0.0
        print(f"    margin [{lo:.2f}, {hi:.2f}): {acc:6.1%} correct  (n={n})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-arch malware-family classifiers.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-per-family", type=int, default=None, help="cap usable samples per family"
    )
    parser.add_argument(
        "--max-entropy", type=float, default=None, help="skip files above this entropy (packed)"
    )
    args = parser.parse_args()

    try:
        datasets = build_dataset(
            max_per_family=args.max_per_family,
            max_entropy=args.max_entropy,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    if not datasets:
        print("no usable samples -- fetch some with `python -m data.fetch_samples` first")
        sys.exit(1)

    for arch, ds in sorted(datasets.items()):
        if not _trainable(ds):
            print(f"\n=== {arch}: {ds.X.shape[0]} samples -- too few per family, skipping ===")
            continue
        # train_test_split can still reject tiny sets (test fold too small for the
        # number of families);
        try:
            model, report = classifier.train(ds.X, ds.y, seed=args.seed)
        except ValueError as e:
            print(f"\n=== {arch}: cannot train -- {e} ===", file=sys.stderr)
            continue
        _print_report(arch, ds, report)
        classifier.save(model, _model_path(arch))
        print(f"  saved -> {_model_path(arch)}")


if __name__ == "__main__":
    main()
