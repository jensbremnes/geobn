"""Disk caching utilities for static geographic data sources.

Cache entries are stored as two files per result:
  {cache_dir}/{hash16}.npy   — float32 numpy array
  {cache_dir}/{hash16}.json  — {"crs": "...", "transform": [a,b,c,d,e,f],
                                "fetched_at": 1758531600.0,
                                "etag": "...", "last_modified": "..."}

``fetched_at`` is the unix time the entry was written or last confirmed
unchanged, used to expire entries older than a source's ``cache_ttl``.
Entries written without it fall back to the array file's modification time.

``etag`` and ``last_modified`` are optional HTTP validators, stored when the
server sent them.  A source that supports conditional requests uses them to
check an expired entry with the server instead of downloading it again.

The 16-char hex hash is SHA-256 of the JSON-serialised cache key dict plus
``_CACHE_VERSION``, so entries written with older semantics are never reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from affine import Affine

from .._types import RasterData

_log = logging.getLogger(__name__)


# Bump when the meaning of cached content changes, so stale entries are
# refetched instead of silently reused.
#   1 — original format
#   2 — declared nodata / masked pixels stored as NaN
_CACHE_VERSION = 2


def _make_cache_path(cache_dir: str | Path, key: dict) -> Path:
    versioned_key = {**key, "_cache_version": _CACHE_VERSION}
    digest = hashlib.sha256(
        json.dumps(versioned_key, sort_keys=True).encode()
    ).hexdigest()[:16]
    return Path(cache_dir).expanduser() / f"{digest}.npy"


def _cache_age(cache_path: Path) -> float | None:
    """Return the age of a cache entry in seconds, or None if it has none.

    The age comes from ``fetched_at`` in the JSON sidecar.  Entries written
    without it fall back to the array file's modification time.
    """
    meta_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not meta_path.exists():
        return None
    fetched_at: float | None = None
    try:
        fetched_at = json.loads(meta_path.read_text()).get("fetched_at")
    except (OSError, ValueError):
        return None
    if fetched_at is None:
        try:
            fetched_at = cache_path.stat().st_mtime
        except OSError:
            return None
    try:
        return max(0.0, time.time() - float(fetched_at))
    except (TypeError, ValueError):
        return None


def _load_cached(cache_path: Path, ttl: float | None = None) -> RasterData | None:
    """Return cached RasterData, or None if absent, corrupt or older than *ttl*.

    Parameters
    ----------
    cache_path:
        Path of the ``.npy`` array file; its ``.json`` sidecar sits alongside.
    ttl:
        Maximum age in seconds.  ``None`` (the default) never expires.
    """
    meta_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not meta_path.exists():
        _log.debug("Cache miss: %s", cache_path.name)
        return None
    if ttl is not None:
        age = _cache_age(cache_path)
        if age is None or age > ttl:
            _log.info("Cache expired: %s", cache_path.name)
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


_VALIDATOR_KEYS = ("etag", "last_modified")


def _clean_validators(validators: dict | None) -> dict:
    """Keep only the known validator keys with non-empty string values."""
    if not validators:
        return {}
    return {
        k: v for k in _VALIDATOR_KEYS
        if isinstance(v := validators.get(k), str) and v
    }


def _load_validators(cache_path: Path) -> dict:
    """Return the HTTP validators stored with a cache entry, or ``{}``."""
    try:
        meta = json.loads(cache_path.with_suffix(".json").read_text())
    except (OSError, ValueError):
        return {}
    return _clean_validators(meta) if isinstance(meta, dict) else {}


def _touch_cached(cache_path: Path, validators: dict | None = None) -> None:
    """Mark an entry as fresh without rewriting its array.

    Used when the server confirms the cached data is unchanged.  Resets
    ``fetched_at`` and replaces the stored validators with *validators*.
    """
    meta_path = cache_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    for k in _VALIDATOR_KEYS:
        meta.pop(k, None)
    meta.update(_clean_validators(validators))
    meta["fetched_at"] = time.time()
    meta_path.write_text(json.dumps(meta))
    _log.info("Cache revalidated: %s", cache_path.name)


def _save_cached(
    cache_path: Path, data: RasterData, validators: dict | None = None
) -> None:
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
        "fetched_at": time.time(),
        **_clean_validators(validators),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta))
