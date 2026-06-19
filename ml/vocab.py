"""Load the mnemonic-root dictionary (``common/categories.json``).

This file does ONE job: read our hand-written dictionary of instruction
"roots" (grouped into semantic categories, one set per CPU architecture) and
hand back the few simple lookups the rest of the pipeline needs:

    roots             ordered list of roots  -> these become the feature columns
    root_index        root  -> its column number
    root_to_category  root  -> the category it belongs to
    categories        category name -> its roots

There is no third-party library that can replace this, because the dictionary
itself is our own domain knowledge (the Mnemocrypt-style category scheme). So
the module stays deliberately tiny: read the JSON, build the lookups, done.

The actual *counting* of instructions into a feature vector is NOT done here --
that is handed to scikit-learn in the Phase-5 feature pipeline, using the
``root_index`` below as a fixed vocabulary.

Usage::

    from ml import vocab
    v = vocab.load("x86-64")     # a parser arch string
    v.num_roots                  # 180
    v.root_index["mov"]          # stable column number
    v.category_of("call")        # "control_flow"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_JSON_PATH = Path(__file__).resolve().parent.parent / "common" / "categories.json"

_ARCH_TO_KEY = {"x86": "x86", "x86-64": "x86", "arm64": "arm64"} # 32 and 64bit architectures share roughly the same mnomonics used for arithmetics/crypto


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Read common/categories.json once and remember it."""
    with _JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh) 


def architectures() -> tuple[str, ...]:
    """The architecture keys in the JSON, e.g. ``("x86", "arm64")``."""
    return tuple(k for k in _raw() if not k.startswith("_")) #filters out metadata


def vocab_key_for_arch(arch: str) -> str | None:
    """Map a parser arch string ('x86'/'x86-64'/'arm64') to a vocab key."""
    return _ARCH_TO_KEY.get(arch)


@dataclass(frozen=True) # This way we can't accidentally change the vocab at runtime
class Vocab:
    """The dictionary for ONE architecture at a time only"""

    arch: str
    roots: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    root_index: dict[str, int]
    root_to_category: dict[str, str] # reverse lookup

    @property
    def num_roots(self) -> int:
        return len(self.roots)

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(self.categories)

    def category_of(self, root: str) -> str | None:
        """The category a root belongs to, or None if it is not in the vocab."""
        return self.root_to_category.get(root)


@lru_cache(maxsize=None)
def load(arch: str) -> Vocab:
    """Load the :class:`Vocab` for a parser arch string or a raw vocab key."""
    key = _ARCH_TO_KEY.get(arch, arch)
    data = _raw()
    if key not in data:
        available = sorted(k for k in data if not k.startswith("_"))
        raise KeyError(f"no vocabulary for {arch!r}; available: {available}") #Error handling in case the feature extractor receives a bin from another architecture other than arm or elf/pe

    categories: dict[str, tuple[str, ...]] = {}
    roots: list[str] = []
    root_to_category: dict[str, str] = {}
    for name, items in data[key].items():
        if name.startswith("_"): # skip metadata
            continue 
        categories[name] = tuple(items)
        for root in items:
            if root in root_to_category:
                raise ValueError( #Dup check
                    f"root {root!r} is in two categories " 
                    f"({root_to_category[root]!r} and {name!r})"
                )
            root_to_category[root] = name
            roots.append(root)

    return Vocab(
        arch=key,
        roots=tuple(roots), #TO REALLY MAKE SURE THE ROOTS ARE UNMMUTABLES
        categories=categories,
        root_index={root: i for i, root in enumerate(roots)}, # give bos h the index and the name of the root, so it indexes automatically
        root_to_category=root_to_category,
    )


if __name__ == "__main__":
    for key in architectures():
        v = load(key)
        print(f"[{key}] categories={v.num_categories} roots={v.num_roots}")
        for cat in v.category_names:
            print(f"    {cat:18s} {len(v.categories[cat]):3d}")
