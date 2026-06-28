"""Load the mnemonic-root dictionary (``common/categories.json``).

This file reads our hand-written dictionary of instruction
"roots" (grouped into semantic categories, one set per CPU architecture) and
hand back the few simple lookups the rest of the pipeline needs

The actual *counting* of instructions into a feature vector is NOT done here,
it'll be handed by scikit later using the ``root_index`` below as a fixed vocabulary.

Usage::

    from ml import vocab
    v = vocab.load("x86-64")     # a parser arch string
    v.num_roots                  # 180
    v.root_index["mov"]          # stable column number
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_JSON_PATH = Path(__file__).resolve().parent.parent / "common" / "categories.json"


_ARCH_TO_KEY = {"x86": "x86", "x86-64": "x86", "arm64": "arm64"}


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Read common/categories.json once and remember it."""
    with _JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Vocab:
    """The dictionary for ONE architecture at a time only"""

    arch: str
    roots: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    root_index: dict[str, int]
    root_to_category: dict[str, str]  # reverse lookup

    @property
    def num_roots(self) -> int:
        return len(self.roots)

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(self.categories)


# Cache the vocab per architecture so we don't rebuild the lookup dicts every time.
@lru_cache(maxsize=None)
def load(arch: str) -> Vocab:
    """Load the :class:`Vocab` for a parser arch string or a raw vocab key."""
    key = _ARCH_TO_KEY.get(arch, arch)  # x86-64 -> x86
    data = _raw()
    if key not in data:
        available = sorted(k for k in data if not k.startswith("_"))
        # a binary from an arch we don't model lands here
        raise ValueError(f"no vocabulary for {key!r}; available: {available}")

    categories: dict[str, tuple[str, ...]] = {}
    roots: list[str] = []
    root_to_category: dict[str, str] = {}
    for name, items in data[key].items():
        if name.startswith("_"):  # skip metadata
            continue
        categories[name] = tuple(items)
        for root in items:
            if root in root_to_category:
                raise ValueError(  # Dup check
                    f"root {root!r} is in two categories "
                    f"({root_to_category[root]!r} and {name!r})"
                )
            root_to_category[root] = name
            roots.append(root)

    return Vocab(
        arch=key,
        roots=tuple(roots),
        categories=categories,
        # give both the index and the name of the root, so it indexes automatically
        root_index={root: i for i, root in enumerate(roots)},
        root_to_category=root_to_category,
    )
