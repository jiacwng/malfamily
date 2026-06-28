"""Turn a parsed binary into a fixed-length numeric feature vector.

This is same idea as the feature extractor from mnemocrypt, but here we use
normalized stats too.
so that a 50-function utility and a 5000-function packer are comparable. We normalize by
the number of *modeled* instructions and track the out-of-vocabulary rate as a diagnostic
that the quality gate uses to drop or abstain on unreadable files (a high OOV rate is one
weak, ambiguous hint that the disassembly may be unreliable).

    mnemonic  --(longest-prefix match)-->  root  --(vocab)-->  category
    counts -> normalized histograms (root-level + category-level)

Public API:
    extract(parsed: ParsedBinary) -> Features"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.parser import ParsedBinary
from ml import vocab


class ExtractorError(RuntimeError):
    """Raised when features cannot be produced for a binary (e.g. an
    architecture we don't have a vocabulary for)."""


@dataclass
class Features:
    """Feature vector for a single binary.

    The vectors are normalized histograms of instruction frequencies in a fixed order,
    so the columns line up perfectly across all samples.
    """
    arch: str
    root_vector: tuple[float, ...]
    category_vector: tuple[float, ...]
    # diagnostics used by the quality gate; NOT part of as_array (the model never sees them)
    total_instructions: int
    mapped_instructions: int  # instructions that matched a root
    # oov = out of vocabulary, rate of instructions whose mnemonic didn't match any root.
    # A high rate can hint at packing/obfuscation, but it is ambiguous on its own.
    oov_rate: float
    num_functions: int

    def as_array(self) -> np.ndarray:
        # called on every binary's Features, then stacked into the training matrix
        return np.asarray(self.root_vector + self.category_vector, dtype=float)


def _root_of(mnemonic: str, roots: frozenset[str]) -> str | None:
    """Map one concrete mnemonic to its vocabulary root, or ``None`` if not in vocabulary."""

    if mnemonic in roots:
        return mnemonic

    # group ARM branches (b.eq, b.ne, etc.) under the b.cond root
    if "b.cond" in roots and mnemonic.startswith("b."):
        return "b.cond"

    # For prefix match, we shrink the mnemonic one char at a time from the end and
    # return the first prefix that is detected
    for end in range(len(mnemonic) - 1, 0, -1):
        prefix = mnemonic[:end]
        if prefix in roots:
            return prefix
    return None


def _count_roots(parsed: ParsedBinary, roots: frozenset[str]) -> tuple[dict[str, int], int, int]:
    """Walk every instruction once, returning (root->count, total, mapped)."""
    counts: dict[str, int] = {}
    # cache/memoization for the _root_of func
    cache: dict[str, str | None] = {}
    total = 0
    mapped = 0
    for fn in parsed.functions:
        for mnemonic in fn.mnemonics:
            total += 1
            if mnemonic in cache:
                root = cache[mnemonic]
            else:
                root = _root_of(mnemonic, roots)
                cache[mnemonic] = root
            if root is None:
                continue
            mapped += 1
            counts[root] = counts.get(root, 0) + 1
    return counts, total, mapped


def _vectorize(
    counts: dict[str, int],
    mapped: int,
    v: vocab.Vocab,  # a vocab dataclass element from vocab.py
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Counts -> (root_vector, category_vector), both normalized by ``mapped``."""

    # vectors initialized
    root_vec = [0.0] * v.num_roots
    cat_names = v.category_names
    cat_index = {name: i for i, name in enumerate(cat_names)}
    cat_vec = [0.0] * len(cat_names)

    for root, count in counts.items():
        frac = count / mapped
        root_vec[v.root_index[root]] = frac

        # take a root, identify which category it belongs to, then add its
        # normalization to this particular category ratio
        cat_vec[cat_index[v.root_to_category[root]]] += frac

    return tuple(root_vec), tuple(cat_vec)


def extract(parsed: ParsedBinary) -> Features:
    """Turn a :class:`ParsedBinary` into a :class:`Features` vector."""

    # Error handling for unsupported architecture
    if parsed.arch.startswith("unsupported:"):
        raise ExtractorError(f"no feature vocabulary for unsupported arch {parsed.arch!r}")
    try:
        v = vocab.load(parsed.arch)  # Map x86-64 to x86 so they share the same vocab.
    except ValueError as e:
        raise ExtractorError(str(e)) from e

    # In order, we fetch the roots lookup catalog, count the instructions from
    # the parsed bin, vectorize the stats
    roots = frozenset(v.roots)  # frozenset makes the set a hashtable to allow O(1) lookups
    counts, total, mapped = _count_roots(parsed, roots)
    root_vector, category_vector = _vectorize(counts, mapped, v)

    # total == 0 means an empty binary (no instructions at all), so OOV is
    # undefined, so we report 0.0 rather than dividing by zero.
    oov_rate = 0.0 if total == 0 else 1.0 - mapped / total

    return Features(
        arch=v.arch,
        root_vector=root_vector,
        category_vector=category_vector,
        total_instructions=total,
        mapped_instructions=mapped,
        oov_rate=oov_rate,
        num_functions=parsed.num_functions,
    )


if __name__ == "__main__":
    import sys

    from core.parser import parse

    for arg in sys.argv[1:]:
        feats = extract(parse(arg))
        print(
            f"{arg}: {feats.arch}  instr={feats.total_instructions} "
            f"mapped={feats.mapped_instructions} oov={feats.oov_rate:.1%}"
        )
        v = vocab.load(feats.arch)
        top = sorted(
            zip(v.category_names, feats.category_vector), key=lambda kv: kv[1], reverse=True
        )
        for name, share in top[:5]:
            print(f"    {name:18s} {share:6.1%}")
