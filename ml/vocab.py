"""Vocabulary loader for malfamily.

Loads ``common/categories.json`` and exposes the mnemonic-root vocabulary in
the shapes the rest of the pipeline needs:

* a stable, ordered flat list of every root (the column order of feature
  vectors depends on this being deterministic);
* the category -> [roots] mapping;
* the reverse root -> category mapping;
* small lookup helpers.

The ordering contract: roots are emitted category-by-category in the order the
categories appear in the JSON file, and within a category in declared order.
This keeps feature-vector columns stable across runs as long as the JSON is
unchanged. Bump ``categories.json``'s ``_meta.version`` if you reorder it so a
stale model is easy to spot.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

# common/categories.json lives one directory up from ml/.
_CATEGORIES_PATH = Path(__file__).resolve().parent.parent / "common" / "categories.json"


def _load_raw() -> Dict[str, object]:
    with _CATEGORIES_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _build() -> Tuple[Dict[str, Tuple[str, ...]], Tuple[str, ...], Dict[str, str]]:
    """Parse the JSON once and cache the derived structures."""
    raw = _load_raw()

    categories: Dict[str, Tuple[str, ...]] = {}
    flat: List[str] = []
    root_to_category: Dict[str, str] = {}

    for name, roots in raw.items():
        if name.startswith("_"):  # skip metadata keys such as "_meta"
            continue
        if not isinstance(roots, list):
            raise ValueError(f"category {name!r} must map to a list, got {type(roots)}")

        cleaned: List[str] = []
        for root in roots:
            if not isinstance(root, str):
                raise ValueError(f"root {root!r} in {name!r} is not a string")
            if root in root_to_category:
                raise ValueError(
                    f"root {root!r} appears in both {root_to_category[root]!r} "
                    f"and {name!r}; roots must be unique across categories"
                )
            root_to_category[root] = name
            cleaned.append(root)
            flat.append(root)

        categories[name] = tuple(cleaned)

    return categories, tuple(flat), root_to_category


def categories() -> Mapping[str, Tuple[str, ...]]:
    """Return the category -> roots mapping (insertion-ordered)."""
    return _build()[0]


def category_names() -> Tuple[str, ...]:
    """Return the category names in declared order."""
    return tuple(_build()[0].keys())


def roots() -> Tuple[str, ...]:
    """Return the full flat list of roots in stable feature-vector order."""
    return _build()[1]


def root_to_category() -> Mapping[str, str]:
    """Return the reverse mapping root -> category name."""
    return _build()[2]


def category_of(root: str) -> str | None:
    """Return the category for a root, or ``None`` if it is unknown."""
    return _build()[2].get(root)


def root_index() -> Mapping[str, int]:
    """Return root -> column index, matching the order of :func:`roots`."""
    return {root: i for i, root in enumerate(_build()[1])}


def num_roots() -> int:
    return len(_build()[1])


def num_categories() -> int:
    return len(_build()[0])


if __name__ == "__main__":
    cats = categories()
    print(f"categories: {num_categories()}")
    print(f"total roots: {num_roots()}")
    for name, rs in cats.items():
        print(f"  {name:18s} {len(rs):3d}")
