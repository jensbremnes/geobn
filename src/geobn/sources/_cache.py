"""Disk caching utilities for static geographic data sources.

Cache entries are stored as two files per result:
  {cache_dir}/{hash16}.npy   — float32 numpy array
  {cache_dir}/{hash16}.json  — {"crs": "...", "transform": [a,b,c,d,e,f]}

The 16-char hex hash is SHA-256 of the JSON-serialised cache key dict.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from affine import Affine

from .._types import RasterData


def _make_cache_path(cache_dir: str | Path, key: dict) -> Path:
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode()
    ).hexdigest()[:16]
    return Path(cache_dir).expanduser() / f"{digest}.npy"


def _load_cached(cache_path: Path) -> RasterData | None:
    """Return cached RasterData or None if absent / corrupt."""
    meta_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        array = np.load(cache_path)
        meta = json.loads(meta_path.read_text())
        transform = Affine(*meta["transform"])
        return RasterData(array=array, crs=meta["crs"], transform=transform)
    except Exception:
        return None  # corrupt → re-fetch


def _save_cached(cache_path: Path, data: RasterData) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data.array)
    meta = {
        "crs": data.crs,
        "transform": [
            data.transform.a, data.transform.b, data.transform.c,
            data.transform.d, data.transform.e, data.transform.f,
        ],
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta))
