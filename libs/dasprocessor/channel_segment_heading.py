#libs\dasprocessor\channel_segment_heading.py
"""
Compute heading (bearing relative to north) and midpoint between two DAS channels.

This module can be imported or run directly for testing.

Functions:
- channel_heading_and_center(channel_a, channel_b, channel_positions)
- annotate_segments_with_heading(doa_json_path, channel_positions_path, output_path)

Example (programmatic):
from dasprocessor.channel_segment_heading import channel_heading_and_center
from dasprocessor.channel_gps import compute_channel_positions

ch_positions = compute_channel_positions("cable-layout.json", 1200, 1.02)
heading, center = channel_heading_and_center(100, 120, ch_positions)
print(heading, center)
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import your channel_gps module
from dasprocessor.channel_gps import compute_channel_positions

# ------------------------------------------------------------

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _rad2deg(x: float) -> float:
    return x * 180.0 / math.pi

def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Bearing from point1 → point2, relative to north, clockwise in degrees.
    Returns 0° = North, 90° = East, 180° = South, 270° = West.
    """
    lat1r, lat2r = _deg2rad(lat1), _deg2rad(lat2)
    dlon = _deg2rad(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    brng = math.atan2(y, x)
    deg = (_rad2deg(brng) + 360.0) % 360.0
    return deg

# ------------------------------------------------------------

def channel_heading_and_center(
    channel_a: int,
    channel_b: int,
    channel_positions: Dict[int, List[float]]
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute the heading (degrees clockwise from north) and the midpoint between two channels.
    Heading is from the lower-numbered channel → higher-numbered channel.

    Returns (heading_deg, (lat_center, lon_center, alt_center))
    """
    ch_low, ch_high = sorted([channel_a, channel_b])
    if ch_low not in channel_positions or ch_high not in channel_positions:
        raise KeyError(f"Missing channel(s) {ch_low} or {ch_high} in provided positions")

    lat1, lon1, alt1 = channel_positions[ch_low]
    lat2, lon2, alt2 = channel_positions[ch_high]

    heading = _bearing_deg(lat1, lon1, lat2, lon2)
    center_lat = (lat1 + lat2) / 2.0
    center_lon = (lon1 + lon2) / 2.0
    center_alt = (alt1 + alt2) / 2.0
    return heading, (center_lat, center_lon, center_alt)

# ------------------------------------------------------------

def annotate_segments_with_heading(
    doa_json_path: str | Path,
    output_path: str | Path
) -> None:
    """
    Reads a DOA JSON (with start_channel/end_channel), adds heading and center_position fields,
    and writes an updated JSON.

    Args
    ----
    doa_json_path : path to the DOA result JSON (with start_channel, end_channel)
    channel_positions_path : path to a JSON with {channel: [lat, lon, alt]}
    output_path : path for output JSON with added headings and centers
    """

    positions = compute_channel_positions(
        geojson_path=r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json",
        channel_count=1200,
        channel_distance=1.02,
        origin="start"  # or "end", "nearest", etc.
    )

    with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_positions.json", "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)

    doa_json_path = Path(doa_json_path)
    channel_positions_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_positions.json")
    output_path = Path(output_path)

    with doa_json_path.open("r", encoding="utf-8") as f:
        doa_data = json.load(f)
    with channel_positions_path.open("r", encoding="utf-8") as f:
        ch_pos = {int(k): v for k, v in json.load(f).items()}

    for packet, pdata in doa_data.items():
        subarrays = pdata.get("subarrays", [])
        for seg in subarrays:
            start_ch = int(seg["start_channel"])
            end_ch = int(seg["end_channel"])
            try:
                heading, center = channel_heading_and_center(start_ch, end_ch, ch_pos)
            except KeyError:
                heading, center = None, (None, None, None)
            seg["heading_relative_to_north"] = heading
            seg["center_position"] = center

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doa_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote annotated DOA JSON with headings to {output_path}")

# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Add geographic headings and center positions to DOA JSON.")
    ap.add_argument("doa_json", type=Path, help="Input DOA JSON (with start_channel, end_channel).")
    ap.add_argument("--output", type=Path, default=Path("doa_with_headings.json"),
                    help="Output JSON path (default: doa_with_headings.json)")
    args = ap.parse_args()

    annotate_segments_with_heading(args.doa_json, args.output)