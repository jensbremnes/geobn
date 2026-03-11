"""Disk caching utilities for static geographic data sources.

Cache entries are stored as two files per result:
  {cache_dir}/{hash16}.npy   — float32 numpy array
  {cache_dir}/{hash16}.json  — {"crs": "...", "transform": [a,b,c,d,e,f]}

The 16-char hex hash is SHA-256 of the JSON-serialised cache key dict.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

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
        _log.debug("Cache miss: %s", cache_path.name)
        return None
    try:
        array = np.load(cache_path)
        meta = json.loads(meta_path.read_text())
        raw = meta["transform"]
        transform = Affine(*raw) if raw is not None else None
        _log.info("Cache hit: %s", cache_path.name)
        return RasterData(array=array, crs=meta["crs"], transform=transform)
    except (OSError, ValueError, KeyError):
        _log.warning("Corrupt cache at %s — will re-fetch", cache_path)
        return None


def _save_cached(cache_path: Path, data: RasterData) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data.array)
    _log.info("Cached to %s", cache_path.name)
    if data.transform is not None:
        t = data.transform
        transform_list: list | None = [t.a, t.b, t.c, t.d, t.e, t.f]
    else:
        transform_list = None
    meta = {
        "crs": data.crs,
        "transform": transform_list,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta))
