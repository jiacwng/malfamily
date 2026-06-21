"""Phase 5: feature pipeline -- manifest + samples -> per-arch (X, y) matrices.

Walks ``data/samples/manifest.csv``, turns each binary into a feature vector
(parse -> extract), and assembles one labeled matrix PER ARCHITECTURE: x86 and
arm64 have different feature dimensions, so they train as separate models.

Extraction is cached per sample (keyed by sha256), so Ghidra 
runs at most once per sample. Re-running after adding samples only parses
the new ones.

NOTE: the cache key is the content hash only. If you change the vocabulary
(``common/categories.json``) or the extractor, delete ``data/feature_cache/`` to
force a rebuild -- otherwise you would train on stale feature vectors.

Usage:
    python -m ml.dataset        # build + print a per-arch / per-family summary
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.extractor import ExtractorError, Features, extract
from core.parser import GhidraError, parse

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SAMPLES_DIR = _DATA_DIR / "samples"
_CACHE_DIR = _DATA_DIR / "feature_cache"


@dataclass
class Dataset:
    """One architecture's labeled feature matrix, ready for a Random Forest."""

    arch: str
    X: np.ndarray  # (n_samples, n_features) float
    y: np.ndarray  # (n_samples,) family labels (str)
    sha256s: list[str]  # row index -> source sample hash, for traceability


def _cache_path(sha256: str, cache_dir: Path) -> Path:
    return cache_dir / f"{sha256}.json"


def _save_features(sha256: str, feats: Features, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(sha256, cache_dir).write_text(
        json.dumps(dataclasses.asdict(feats)), encoding="utf-8"
    )


def _load_features(sha256: str, cache_dir: Path) -> Features:
    d = json.loads(_cache_path(sha256, cache_dir).read_text(encoding="utf-8"))
    # since json loads returns lists rather than tuples, we have to reconvert them into tuple after reloading
    d["root_vector"] = tuple(d["root_vector"])
    d["category_vector"] = tuple(d["category_vector"])
    return Features(**d)


def featurize(path: Path, sha256: str, cache_dir: Path = _CACHE_DIR) -> Features:
    """Features for one sample and caches it if not already in cache"""
    if _cache_path(sha256, cache_dir).exists():
        return _load_features(sha256, cache_dir)
    feats = extract(parse(path))  
    _save_features(sha256, feats, cache_dir)
    return feats


def _read_manifest(samples_dir: Path) -> list[tuple[str, str]]:
    """(sha256, family) rows from the downloader's manifest."""
    manifest = samples_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest} -- run `python -m data.fetch_samples` first"
        )
    with manifest.open(newline="", encoding="utf-8") as fh:
        return [(row["sha256"], row["family"]) for row in csv.DictReader(fh)]


def build_dataset(
    samples_dir: Path = _SAMPLES_DIR, cache_dir: Path = _CACHE_DIR
) -> dict[str, Dataset]:
    """Assemble one :class:`Dataset` per architecture from all downloaded samples.

    Samples that can't be parsed (Ghidra failure) or whose architecture 
    we don't model are logged and skipped.
    """
    # accumulator, a dictionary keyed by architecture (architecture, hashes)
    buckets: dict[str, tuple[list[np.ndarray], list[str], list[str]]] = {}
    # To avoid leakage during the train/test split
    seen: set[str] = set()
    for sha256, family in _read_manifest(samples_dir):
        if sha256 in seen:
            continue
        seen.add(sha256)
        path = samples_dir / family / f"{sha256}.bin"
        if not path.is_file():
            print(f"  skip {sha256[:12]}..: file not found", file=sys.stderr)
            continue
        try:
            feats = featurize(path, sha256, cache_dir)
        except (GhidraError, ExtractorError) as e:
            print(f"  skip {sha256[:12]}..: {e}", file=sys.stderr)
            continue
        vecs, fams, shas = buckets.setdefault(feats.arch, ([], [], [])) # for each sample in the manifest that is unique, featurize it
        # and file its (vec,label,hash) under its arch
        vecs.append(feats.as_array())
        fams.append(family)
        shas.append(sha256)

    return {
        arch: Dataset(arch, np.vstack(vecs), np.array(fams), shas)
        for arch, (vecs, fams, shas) in buckets.items()
    }


if __name__ == "__main__":
    from collections import Counter

    try:
        datasets = build_dataset()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not datasets:
        print("no usable samples -- fetch some with `python -m data.fetch_samples` first")
    for arch, ds in sorted(datasets.items()):
        print(f"[{arch}] {ds.X.shape[0]} samples x {ds.X.shape[1]} features")
        for fam, n in sorted(Counter(ds.y.tolist()).items()):
            print(f"    {fam:24s} {n}")
