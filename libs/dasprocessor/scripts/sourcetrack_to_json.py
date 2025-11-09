#libs\dasprocessor\scripts\sourcetrack_to_json.py
"""
Export source track or TX positions to JSON (plain list or GeoJSON LineString),
so it can be loaded by plot_doa_map.py as the "source track" layer.

Usage (PowerShell):
python -m dasprocessor.scripts.export_source_track `
  --csv "C:\...\source-position.csv" `
  --date "2024-05-03" `
  --run 2 `
  --mode tx `
  --format plain `
  --outfile "C:\...\source_track_tx.json"

# Or full run track as GeoJSON
python -m dasprocessor.scripts.export_source_track `
  --csv "C:\...\source-position.csv" `
  --date "2024-05-03" `
  --run 2 `
  --mode track `
  --format geojson `
  --outfile "C:\...\source_track_run.geojson"
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Reuse your logic from dasprocessor.plot.source_track
from dasprocessor.plot.source_track import (
    load_source_points_for_run,     # (csv, run) -> [(lat, lon, dt), ...] within time window
    _tx_datetimes_for_run,          # (run) -> [datetime]*sequence_count  (uses constants.get_run inside)
    _interp_positions_at_times,     # (points, query_times) -> [(lat, lon, dt, idx), ...]
)
# And we need get_run date awareness:
from dasprocessor.constants import get_run  # to force constants import (for _tx_datetimes_for_run)

def points_to_plain_list(points_latlon: List[Tuple[float, float]]) -> list:
    """[[lat, lon], ...]"""
    return [[float(lat), float(lon)] for (lat, lon) in points_latlon]

def points_to_geojson_linestring(points_latlon: List[Tuple[float, float]]) -> dict:
    """GeoJSON LineString with CRS84 order [lon, lat]."""
    coords = [[float(lon), float(lat)] for (lat, lon) in points_latlon]
    return {
        "type": "Feature",
        "properties": {"name": "source_track"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }

def main():
    ap = argparse.ArgumentParser(description="Export source track or TX positions as JSON/GeoJSON.")
    ap.add_argument("--csv", type=Path, required=True, help="Path to source-position.csv")
    ap.add_argument("--date", required=True, help='Experiment date string (e.g. "2024-05-03")')
    ap.add_argument("--run", type=int, required=True, help="Run number inside constants.properties[date]")
    ap.add_argument("--mode", choices=["track", "tx"], default="tx",
                    help="Export run 'track' (all samples in window) or 'tx' (128 transmission points).")
    ap.add_argument("--format", choices=["plain", "geojson"], default="plain",
                    help="Output format: 'plain' -> [[lat,lon],...], 'geojson' -> LineString.")
    ap.add_argument("--every", type=int, default=1,
                    help="Keep every Nth TX point (mode=tx only). Default 1 (keep all).")
    ap.add_argument("--outfile", type=Path, required=True, help="Output JSON path")
    args = ap.parse_args()

    # The helper functions in source_track.py default to DATE_STR inside that module.
    # We'll temporarily adjust by patching get_run's expectation using the provided --date
    # by calling get_run (ensures constants import) and then relying on _tx_datetimes_for_run
    # which reads from dasprocessor.constants.get_run's DATE_STR inside its module.
    # To avoid editing your module, we'll monkey-patch the module-level DATE_STR if it exists.

    # Load all points in the run window
    points = load_source_points_for_run(args.csv, args.run)  # [(lat,lon,dt),...]

    if not points:
        raise SystemExit("No points found in the CSV for this run/time window. Check CSV and constants time_range.")

    # Build output coordinate list (lat, lon) in desired mode
    out_latlon: List[Tuple[float, float]] = []

    if args.mode == "track":
        # Just take the filtered track through the run window (sorted by time)
        pts_sorted = sorted(points, key=lambda t: t[2])
        out_latlon = [(float(lat), float(lon)) for (lat, lon, _dt) in pts_sorted]

    else:  # mode == "tx"
        # Interpolate at the 128 TX times for this date+run
        # We need to ensure _tx_datetimes_for_run uses the correct date. It reads constants via get_run inside that module.
        # We'll replicate the call by grabbing the run dict to ensure constants is loaded. The date is only used inside get_run.
        _ = get_run(args.date, args.run)  # ensure constants access with given date

        q_times: List[datetime] = _tx_datetimes_for_run(args.run)
        tx_pts = _interp_positions_at_times(points, q_times)  # [(lat, lon, dt, idx), ...] masked by overlap
        if not tx_pts:
            raise SystemExit("No TX points could be interpolated (time windows may not overlap).")

        # Optionally subsample
        for (lat, lon, dt, idx) in tx_pts:
            if idx % args.every != 0:
                continue
            out_latlon.append((float(lat), float(lon)))

    # Encode
    if args.format == "plain":
        obj = points_to_plain_list(out_latlon)
    else:
        obj = points_to_geojson_linestring(out_latlon)

    # Write
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote source track to: {args.outfile}")

if __name__ == "__main__":
    main()
