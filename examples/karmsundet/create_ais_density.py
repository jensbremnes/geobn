"""Create AIS traffic density raster for Karmsundet.

This script downloads raw AIS position data from Kystverket (Norwegian Coastal
Administration), filters it to the Karmsundet study area, and bins it into a
GeoTIFF matching the grid used by ``run_example.py``.

Data source
-----------
Kystverket makes historical AIS data freely available via their portal:
    https://kystdatahuset.no

Download steps
--------------
1. Go to https://kystdatahuset.no and navigate to AIS Data > Historiske data.
2. Select a date range (e.g. 30 days of a recent month).
3. Set the bounding box:  WEST=5.15  SOUTH=59.25  EAST=5.55  NORTH=59.55
4. Download the CSV export.  The file name typically looks like:
       aisdata_<date>_<date>.csv
5. Place the CSV file in the same directory as this script or pass its path
   as a command-line argument:
       uv run python examples/karmsundet/create_ais_density.py aisdata.csv

CSV format (Kystverket standard)
---------------------------------
The CSV is semicolon-delimited with at least these columns (column names may
vary slightly between export versions):

    timestamp   — ISO-8601 date-time, e.g. 2024-06-15T12:34:56Z
    mmsi        — Maritime Mobile Service Identity (vessel ID)
    lat         — WGS84 latitude
    lon         — WGS84 longitude
    sog         — Speed over ground (knots)
    cog         — Course over ground (degrees)
    nav_status  — AIS navigational status code

Processing pipeline
-------------------
1. Read CSV, parse lat/lon/sog.
2. Filter to bounding box + sog > 0.5 knots (remove anchored vessels).
3. Bin positions into the same pixel grid as run_example.py.
4. Normalise to encounters/km²/day.
5. Write single-band float32 GeoTIFF with rasterio.

Output
------
    examples/karmsundet/data/ais_density_karmsundet.tif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Grid constants — must match run_example.py exactly
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 5.15, 59.25, 5.55, 59.55
RESOLUTION = 0.002   # degrees  →  150 rows × 200 cols

HERE     = Path(__file__).parent
OUT_PATH = HERE / "data" / "ais_density_karmsundet.tif"


def _pixel_area_km2() -> float:
    """Approximate pixel area in km² at the study area centre latitude."""
    lat_mid = (SOUTH + NORTH) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_mid))
    dy_m = RESOLUTION * m_per_deg_lat
    dx_m = RESOLUTION * m_per_deg_lon
    return (dx_m * dy_m) / 1e6


def process_ais_csv(csv_path: Path, n_days: int = 30) -> None:
    """Read Kystverket AIS CSV and write a density GeoTIFF.

    Parameters
    ----------
    csv_path:
        Path to the downloaded AIS CSV file.
    n_days:
        Number of days covered by the CSV export (used to normalise to
        encounters/km²/day).  Default: 30.
    """
    import rasterio
    from affine import Affine

    print(f"Reading AIS data from {csv_path} ...")

    # Try common Kystverket separator variants
    try:
        import csv as csv_mod
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
    except Exception as exc:
        sys.exit(f"Cannot read CSV: {exc}")

    # Parse with numpy genfromtxt for speed; skip header row
    try:
        raw = np.genfromtxt(
            csv_path,
            delimiter=delimiter,
            names=True,
            dtype=None,
            encoding="utf-8-sig",
            invalid_raise=False,
        )
    except Exception as exc:
        sys.exit(f"Failed to parse CSV: {exc}")

    # Normalise column names (lower-case, strip whitespace)
    col_map = {n.lower().strip(): n for n in raw.dtype.names}

    def _get(candidates):
        for c in candidates:
            if c in col_map:
                return raw[col_map[c]].astype(float)
        available = list(col_map.keys())
        sys.exit(f"Cannot find column (tried {candidates}). Available: {available}")

    lats = _get(["lat", "latitude", "lat_deg"])
    lons = _get(["lon", "longitude", "lon_deg"])
    sogs = _get(["sog", "speed_over_ground", "speed"])

    # Filter: bounding box + moving vessels
    mask = (
        (lats >= SOUTH) & (lats <= NORTH) &
        (lons >= WEST)  & (lons <= EAST)  &
        (sogs > 0.5)
    )
    lats = lats[mask]
    lons = lons[mask]
    print(f"Filtered     : {mask.sum():,} positions within bbox and sog > 0.5 kts")

    if mask.sum() == 0:
        sys.exit(
            "No AIS positions passed the filter.  Check that the CSV covers "
            "the study area and that column names match expected variants."
        )

    # Bin into grid
    H = round((NORTH - SOUTH) / RESOLUTION)
    W = round((EAST  - WEST)  / RESOLUTION)

    row_idx = np.clip(
        ((NORTH - lats) / RESOLUTION).astype(int), 0, H - 1
    )
    col_idx = np.clip(
        ((lons - WEST)  / RESOLUTION).astype(int), 0, W - 1
    )

    counts = np.zeros((H, W), dtype=np.float32)
    np.add.at(counts, (row_idx, col_idx), 1)

    # Normalise to encounters/km²/day
    area_km2 = _pixel_area_km2()
    density  = counts / (area_km2 * n_days)

    print(f"Density range: {density.min():.4f} – {density.max():.4f} enc/km²/day")
    print(f"Non-zero pixels: {(density > 0).sum():,} / {H * W:,}")

    # Write GeoTIFF
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(RESOLUTION, 0, WEST, 0, -RESOLUTION, NORTH)

    profile = dict(
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=float("nan"),
    )
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(density, 1)

    print(f"Written → {OUT_PATH}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: uv run python create_ais_density.py <aisdata.csv> [n_days]")
        sys.exit(0)

    csv_path = Path(sys.argv[1])
    n_days   = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    process_ais_csv(csv_path, n_days=n_days)


if __name__ == "__main__":
    main()
