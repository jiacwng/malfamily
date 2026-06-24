"""feature pipeline, manifest + samples -> per-arch (X, y) matrices.

Walks ``data/samples/manifest.csv``, turns each binary into a feature vector
(parse -> extract), and assembles one labeled matrix PER ARCHITECTURE: x86 and
arm64 have different feature dimensions, so they train as separate models.

Extraction is cached per sample (keyed by sha256), so Ghidra
runs at most once per sample. Re-running after adding samples only parses
the new ones.

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

_MAX_OOV_RATE = 0.5
_MIN_MAPPED = 200


@dataclass
class Dataset:
    """One architecture's labeled feature matrix, ready for a Random Forest."""

    arch: str
    X: np.ndarray  # (n_samples, n_features) float
    y: np.ndarray  # (n_samples,) family labels (str)
    sha256s: list[str]  # row index -> source sample hash, for traceability


def _file_entropy(path: Path) -> float:
    """Shannon entropy in bits/byte (0-8 lowest to highest)."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:  # avoids divide by 0 in the fraction
        return 0.0
    counts = np.bincount(data, minlength=256)
    p = counts[counts > 0] / data.size
    return float(-(p * np.log2(p)).sum())


def _cache_path(sha256: str, cache_dir: Path) -> Path:
    return cache_dir / f"{sha256}.json"


def _save_features(sha256: str, feats: Features, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(sha256, cache_dir).write_text(
        json.dumps(dataclasses.asdict(feats)), encoding="utf-8"
    )


def _load_features(sha256: str, cache_dir: Path) -> Features:
    d = json.loads(_cache_path(sha256, cache_dir).read_text(encoding="utf-8"))
    # since json loads returns lists rather than tuples, we have to reconvert
    # them into tuple after reloading
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
    # utf-8-sig tolerates a UTF-8 BOM (e.g. if the manifest was re-saved by a
    # tool like PowerShell's Export-Csv) so the first column key stays "sha256".
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        return [(row["sha256"], row["family"]) for row in csv.DictReader(fh)]


def build_dataset(
    samples_dir: Path = _SAMPLES_DIR,
    cache_dir: Path = _CACHE_DIR,
    max_oov_rate: float = _MAX_OOV_RATE,
    min_mapped: int = _MIN_MAPPED,
    max_per_family: int | None = None,
    max_entropy: float | None = None,
) -> dict[str, Dataset]:
    """Assemble one :class:`Dataset` per architecture from all downloaded samples.

    Samples that can't be parsed (Ghidra failure), whose architecture we don't
    model, or that fail the quality gate (too packed / too little code recovered)
    are logged and skipped. We added a max family to balance out the data despite
    a balance being made
    """
    # accumulator, a dictionary keyed by architecture (architecture, hashes)
    buckets: dict[str, tuple[list[np.ndarray], list[str], list[str]]] = {}
    # To avoid leakage during the train/test split
    seen: set[str] = set()
    kept: dict[str, int] = {}  # usable samples kept per family, for max_per_family
    rows = _read_manifest(samples_dir)
    # progress to stdout (flush=True so it shows live during slow Ghidra parses)
    print(f"building dataset from {len(rows)} manifest rows...", flush=True)
    for i, (sha256, family) in enumerate(rows, 1):
        if sha256 in seen:
            continue
        seen.add(sha256)
        # family cap: once a family has enough usable samples, skip the rest
        if max_per_family is not None and kept.get(family, 0) >= max_per_family:
            continue
        path = samples_dir / family / f"{sha256}.bin"
        if not path.is_file():
            print(f"  [{i}/{len(rows)}] {sha256[:12]}.. skip: file not found", flush=True)
            continue
        cached = _cache_path(sha256, cache_dir).exists()
        # filter to keep on the uncached and files whose entrepy is below the threshold
        # (likely not packed)
        if not cached and max_entropy is not None:
            ent = _file_entropy(path)
            if ent > max_entropy:
                print(
                    f"  [{i}/{len(rows)}] {sha256[:12]}.. ({family}) "
                    f"skip: likely packed (entropy {ent:.2f})",
                    flush=True,
                )
                continue
        step = "cached" if cached else "parsing via ghidra (can take minutes)..."
        print(f"  [{i}/{len(rows)}] {sha256[:12]}.. ({family}) {step}", flush=True)
        try:
            feats = featurize(path, sha256, cache_dir)
        except (GhidraError, ExtractorError) as e:
            print(f"      skip: {e}", flush=True)
            continue
        # quality threshold
        if feats.oov_rate > max_oov_rate or feats.mapped_instructions < min_mapped:
            print(
                f"      skip (low quality): {feats.mapped_instructions} mapped, "
                f"{feats.oov_rate:.0%} OOV",
                flush=True,
            )
            continue
        vecs, fams, shas = buckets.setdefault(feats.arch, ([], [], []))
        vecs.append(feats.as_array())
        fams.append(family)
        shas.append(sha256)
        kept[family] = kept.get(family, 0) + 1
        print(
            f"      -> {feats.arch}, {feats.num_functions} funcs, "
            f"{feats.mapped_instructions} mapped, {feats.oov_rate:.0%} OOV",
            flush=True,
        )

    return {
        arch: Dataset(arch, np.vstack(vecs), np.array(fams), shas)
        for arch, (vecs, fams, shas) in buckets.items()
    }


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Build the per-arch feature datasets.")
    parser.add_argument(
        "--max-per-family", type=int, default=None, help="cap usable samples per family"
    )
    parser.add_argument(
        "--max-entropy", type=float, default=None, help="skip files above this entropy (packed)"
    )
    args = parser.parse_args()

    try:
        datasets = build_dataset(max_per_family=args.max_per_family, max_entropy=args.max_entropy)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not datasets:
        print("no usable samples -- fetch some with `python -m data.fetch_samples` first")
    for arch, ds in sorted(datasets.items()):
        print(f"[{arch}] {ds.X.shape[0]} samples x {ds.X.shape[1]} features")
        for fam, n in sorted(Counter(ds.y.tolist()).items()):
            print(f"    {fam:24s} {n}")
