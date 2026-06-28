"""Classify one binary file against the per-architecture model.

If a file is packed we don't drop it (like in training) -- we return a prediction
flagged as 'unanalyzable' with the evidence.

Public API:
    classify(path) -> Prediction
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.extractor import Features, extract
from core.parser import GhidraError, parse
from ml import classifier

_MODEL_DIR = Path(__file__).resolve().parent
_MAX_OOV_RATE = 0.5  # too many unknown instructions means it's packed
_MIN_MAPPED = 200  # too few instructions means we only got the packer stub
_MIN_MARGIN = 0.25  # if the top 2 predictions are within 25%, flag as low confidence


class PredictError(RuntimeError):
    """No trained model exists for the binary's architecture."""


@dataclass
class Prediction:
    arch: str
    label: str | None  # predicted malware type, or None when unanalyzable
    confidence: float
    margin: float
    status: str
    oov_rate: float
    mapped_instructions: int
    entropy: float = 0.0  # whole-file Shannon entropy, reported for packed files
    runner_up: str | None = None  # 2nd-best type, to show how close the call was
    runner_up_confidence: float = 0.0


def _file_entropy(path: str | Path) -> float:
    """Whole-file Shannon entropy -- a rough packing signal shown in the report."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return 0.0
    counts = np.bincount(data, minlength=256)
    p = counts[counts > 0] / data.size
    return float(-(p * np.log2(p)).sum())


def _predict_from_features(
    feats: Features, model_dir: Path, *, min_margin: float = _MIN_MARGIN, entropy: float = 0.0
) -> Prediction:
    # quality check: a packed binary is flagged unanalyzable, never guessed
    if feats.mapped_instructions < _MIN_MAPPED or feats.oov_rate > _MAX_OOV_RATE:
        return Prediction(
            arch=feats.arch,
            label=None,
            confidence=0.0,
            margin=0.0,
            status="unanalyzable",
            oov_rate=feats.oov_rate,
            mapped_instructions=feats.mapped_instructions,
            entropy=entropy,
        )

    model_path = model_dir / f"model_{feats.arch}.pkl"
    if not model_path.exists():
        raise PredictError(f"no trained model for arch {feats.arch!r} (expected {model_path})")
    model = classifier.load(model_path)

    label, confidence, margin, runner_up, runner_up_conf = classifier.predict(
        model, feats.as_array()
    )
    status = "ok" if margin >= min_margin else "low_confidence"
    return Prediction(
        arch=feats.arch,
        label=label,
        confidence=confidence,
        margin=margin,
        status=status,
        oov_rate=feats.oov_rate,
        mapped_instructions=feats.mapped_instructions,
        entropy=entropy,
        runner_up=runner_up,
        runner_up_confidence=runner_up_conf,
    )


def classify(path: str | Path, model_dir: Path = _MODEL_DIR) -> Prediction:
    """parse + extract + classify one binary."""
    feats = extract(parse(path))
    return _predict_from_features(feats, model_dir, entropy=_file_entropy(path))


def _format(name: str, p: Prediction) -> str:
    """One human-readable line per binary for the CLI."""
    if p.status == "unanalyzable":
        return (
            f"{name}: UNANALYZABLE (likely packed) -- {p.arch}, "
            f"entropy {p.entropy:.2f}, {p.oov_rate:.0%} Out of Vocabulary, "
            f"{p.mapped_instructions} instr"
        )
    runner = f", 2nd: {p.runner_up} ({p.runner_up_confidence:.0%})" if p.runner_up else ""
    flag = "" if p.status == "ok" else f"  [LOW CONFIDENCE, margin {p.margin:.2f}]"
    return f"{name}: {p.label} ({p.confidence:.0%}){runner}{flag} -- {p.arch}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify binaries by malware type.")
    parser.add_argument("binaries", nargs="+", help="path(s) to PE/ELF/Mach-O files")
    args = parser.parse_args()

    for b in args.binaries:
        try:
            print(_format(b, classify(b)))
        except (GhidraError, PredictError, FileNotFoundError) as e:
            print(f"{b}: error -- {e}", file=sys.stderr)
