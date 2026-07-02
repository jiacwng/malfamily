"""
Loads the manifest, extracts features from the binaries, and groups them by architecture.

Features are cached by SHA256 to avoid running Ghidra multiple times on the same sample
Usage:
    python -m ml.dataset        # build + print a per-arch / per-type summary
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from collections import Counter
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

# Maps folder names on disk (malware families) to our 6 training labels.
# We simplified from 8 labels to 6 to help our model learn better:
# - Merged Backdoor into RAT and Dropper into Loader because they did similar things
#   and confused the model.
# - New families were found using tags, but we still label them by family name because
#   tags are too noisy.
# - Removed Adware/Rootkit (not enough files) and Mirai (CPU architectures we don't support).
FAMILY_TO_TYPE = {
    # Infostealer
    "Vidar": "Infostealer",
    "AmosStealer": "Infostealer",  # macOS (Objective-See)
    "RedLineStealer": "Infostealer",
    "AgentTesla": "Infostealer",
    "Formbook": "Infostealer",
    "Cuckoo": "Infostealer",  # macOS (Objective-See)
    "AtomicStealer": "Infostealer",  # macOS (Objective-See)
    "KeySteal": "Infostealer",  # macOS (Objective-See)
    "SalatStealer": "Infostealer",  # tag-discovery
    "ScarfaceStealer": "Infostealer",  # tag-discovery
    "RemusStealer": "Infostealer",  # tag-discovery
    "Rhadamanthys": "Infostealer",  # tag-discovery
    # RAT (absorbs the former Backdoor class)
    "RemcosRAT": "RAT",
    "AsyncRAT": "RAT",
    "Gh0stRAT": "RAT",
    "NetWire": "RAT",
    "DarkComet": "RAT",
    "RustBucket": "RAT",  # macOS (Objective-See)
    "OceanLotus": "RAT",  # macOS (Objective-See); was Backdoor
    "CobaltStrike": "RAT",  # was Backdoor
    "ValleyRAT": "RAT",  # tag-discovery (native C++ -- parses well)
    # NanoCore / QuasarRAT / DCRat / njrat dropped: pure .NET (MSIL), so Ghidra
    # sees only a 1-instruction native stub -> all skip the quality gate (or hang).
    # A native-code classifier can't use managed-code families.
    # Ransomware
    "Stop": "Ransomware",
    "LockBit": "Ransomware",
    "EvilQuest": "Ransomware",  # macOS (Objective-See)
    "KeRanger": "Ransomware",  # macOS (Objective-See)
    "macRansom": "Ransomware",  # macOS (Objective-See)
    "WannaCry": "Ransomware",
    "Akira": "Ransomware",  # tag-discovery
    "RAWorld": "Ransomware",  # tag-discovery
    "Qilin": "Ransomware",  # tag-discovery
    "BianLian": "Ransomware",  # tag-discovery
    # Loader (absorbs the former Dropper class)
    "Smoke Loader": "Loader",
    "Amadey": "Loader",
    "Emotet": "Loader",
    "GuLoader": "Loader",  # was Dropper
    "Shlayer": "Loader",  # macOS (Objective-See); was Dropper
    "Pikabot": "Loader",  # was Backdoor
    "Matanbuchus": "Loader",  # tag-discovery
    "Wslink": "Loader",  # tag-discovery
    "PrivateLoader": "Loader",  # tag-discovery
    # Banker
    "Dridex": "Banker",
    "IcedID": "Banker",
    "TrickBot": "Banker",
    "Mekotio": "Banker",  # tag-discovery
    "Grandoreiro": "Banker",  # tag-discovery
    "Ousaban": "Banker",  # tag-discovery
    # Miner
    "CoinMiner": "Miner",
    "BirdMiner": "Miner",  # macOS (Objective-See)
    "SilentCryptoMiner": "Miner",  # tag-discovery
}


@dataclass
class Dataset:
    """One architecture's labeled feature matrix, ready for a Random Forest."""

    arch: str
    X: np.ndarray  # (n_samples, n_features) float
    y: np.ndarray  # (n_samples,) malware-type labels (str)
    sha256s: list[str]  # Keeps track of which file hash belongs to each row in the dataset.


def _file_entropy(path: Path) -> float:
    """Shannon entropy"""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
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


def _save_failure(sha256: str, reason: str, cache_dir: Path) -> None:
    """Negative cache: remember a *deterministic* extraction failure (e.g. an
    unsupported architecture) so we never re-run Ghidra on this sample again."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(sha256, cache_dir).write_text(
        json.dumps({"extractor_error": reason}), encoding="utf-8"
    )


def _load_features(sha256: str, cache_dir: Path) -> Features:
    d = json.loads(_cache_path(sha256, cache_dir).read_text(encoding="utf-8"))
    # Cached failure: We throw the error immediately to skip running Ghidra again on
    # stuff we already know will fail.
    if "extractor_error" in d:
        raise ExtractorError(d["extractor_error"])
    # Since json loads returns lists rather than tuples, we have to reconvert
    # them into tuple after reloading
    d["root_vector"] = tuple(d["root_vector"])
    d["category_vector"] = tuple(d["category_vector"])
    return Features(**d)


def featurize(path: Path, sha256: str, cache_dir: Path = _CACHE_DIR) -> Features:
    """Features for one sample, cached by SHA256.

    A successful extraction caches its feature vector; a deterministic failure
    (unsupported arch) is negative-cached so Ghidra never re-runs on it. Transient
    failures (GhidraError timeouts) are left uncached so they can be retried.
    """
    if _cache_path(sha256, cache_dir).exists():
        return _load_features(sha256, cache_dir)
    try:
        feats = extract(parse(path))
    except ExtractorError as e:
        _save_failure(sha256, str(e), cache_dir)
        raise
    _save_features(sha256, feats, cache_dir)
    return feats


def _read_manifest(samples_dir: Path, manifest_name: str = "manifest.csv") -> list[tuple[str, str]]:
    """(sha256, family) rows from the downloader's manifest."""
    manifest = samples_dir / manifest_name
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest} (it ships with the repo alongside the feature bundle)"
        )
    # Use utf-8-sig so hidden formatting characters
    # from Excel/PowerShell CSVs don't mess up the first column header.
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        return [(row["sha256"], row["family"]) for row in csv.DictReader(fh)]


def build_dataset(
    samples_dir: Path = _SAMPLES_DIR,
    cache_dir: Path = _CACHE_DIR,
    max_oov_rate: float = _MAX_OOV_RATE,
    min_mapped: int = _MIN_MAPPED,
    max_entropy: float | None = None,
    manifest_name: str = "manifest.csv",
) -> dict[str, Dataset]:
    """Assemble one :class:`Dataset` per architecture from all downloaded samples."""
    # Accumulator, a dictionary keyed by architecture (architecture, hashes)
    buckets: dict[str, tuple[list[np.ndarray], list[str], list[str]]] = {}
    # To avoid leakage/duplicated samples during the train/test split
    seen: set[str] = set()
    rows = _read_manifest(samples_dir, manifest_name)
    # progress to stdout (flush=True so it shows live during slow Ghidra parses)
    print(f"building dataset from {len(rows)} manifest rows...", flush=True)
    for i, (sha256, family) in enumerate(rows, 1):
        if sha256 in seen:
            continue
        seen.add(sha256)
        # We use malware type as the label, and ignore any family that isn't mapped.
        category = FAMILY_TO_TYPE.get(family)
        if category is None:
            continue
        cached = _cache_path(sha256, cache_dir).exists()
        path = samples_dir / family / f"{sha256}.bin"

        if not cached:
            if not path.is_file():
                print(f"  [{i}/{len(rows)}] {sha256[:12]}.. skip: not cached, no .bin", flush=True)
                continue
            # skip likely-packed files before paying for a Ghidra run
            if max_entropy is not None and (ent := _file_entropy(path)) > max_entropy:
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
        vecs, types, shas = buckets.setdefault(feats.arch, ([], [], []))
        vecs.append(feats.as_array())
        types.append(category)
        shas.append(sha256)
        print(
            f"      -> {feats.arch}, {feats.num_functions} funcs, "
            f"{feats.mapped_instructions} mapped, {feats.oov_rate:.0%} OOV",
            flush=True,
        )

    return {
        arch: Dataset(arch, np.vstack(vecs), np.array(types), shas)
        for arch, (vecs, types, shas) in buckets.items()
    }


# Code to generate the train-ready distribution artifact. Creates a compressed .npz archive 
# per architecture containing validated feature matrices (float64 for deterministic reproducibility), 
# labels, and source hashes.
_BUNDLE_PREFIX = "features_"


def export_bundle(datasets: dict[str, Dataset], out_dir: Path = _DATA_DIR) -> list[Path]:
    """Write each arch's (X, y, sha) to data/features_<arch>.npz and return the paths.

    Arches too small to train (a single class, or any class with one sample, which would
    break the stratified split) are skipped, so a stray off-arch sample never ships.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arch, ds in datasets.items():
        label_counts = Counter(ds.y.tolist())
        if len(label_counts) < 2 or min(label_counts.values()) < 2:
            print(f"skip bundle for {arch}: too small to train ({dict(label_counts)})")
            continue
        path = out_dir / f"{_BUNDLE_PREFIX}{arch}.npz"
        np.savez_compressed(
            path, X=ds.X.astype(np.float64), y=ds.y, sha=np.asarray(ds.sha256s)
        )
        written.append(path)
    return written


def load_bundle(bundle_dir: Path = _DATA_DIR) -> dict[str, Dataset]:
    """Load shipped feature bundles into Datasets, or {} when none exist (so callers
    can fall back to building from the cache)."""
    out: dict[str, Dataset] = {}
    for path in sorted(bundle_dir.glob(f"{_BUNDLE_PREFIX}*.npz")):
        arch = path.stem[len(_BUNDLE_PREFIX):]
        with np.load(path, allow_pickle=False) as d:
            out[arch] = Dataset(arch, d["X"], d["y"].astype(str), list(d["sha"].astype(str)))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the per-arch feature datasets.")
    parser.add_argument(
        "--max-entropy", type=float, default=None, help="skip files above this entropy (packed)"
    )
    parser.add_argument(
        "--manifest", default="manifest.csv", help="manifest filename under data/samples/"
    )
    parser.add_argument(
        "--export-bundle", action="store_true",
        help="write data/features_<arch>.npz for shipping (train-on-clone artifact)"
    )
    args = parser.parse_args()

    try:
        datasets = build_dataset(max_entropy=args.max_entropy, manifest_name=args.manifest)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not datasets:
        print("no usable samples -- fetch some with `python -m data.fetch_samples` first")
    for arch, ds in sorted(datasets.items()):
        print(f"[{arch}] {ds.X.shape[0]} samples x {ds.X.shape[1]} features")
        for fam, n in sorted(Counter(ds.y.tolist()).items()):
            print(f"    {fam:24s} {n}")

    if args.export_bundle:
        for path in export_bundle(datasets):
            print(f"wrote bundle -> {path}")
