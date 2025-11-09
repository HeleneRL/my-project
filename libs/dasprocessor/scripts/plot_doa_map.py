#libs\dasprocessor\scripts\plot_doa_map.py
"""
Plot DOA/segments + channels + source track on a Folium map.

Usage (PowerShell):
python -m dasprocessor.scripts.plot_doa_map `
  --doa-json "C:\...\packet_doa_with_headings.json" `
  --channel-positions "C:\...\channel_positions.json" `
  --source-track "C:\...\source_track.json" `
  --outfile "C:\...\doa_map.html" `
  --doa-ray-length-m 300

Layers:
- Channels (markers + light polyline)
- Source track (polyline)
- Array segments (polyline between first/last channel)
- DOA rays (from segment center, using actual DOA azimuth)

IMPORTANT about DOA azimuth:
We compute "actual DOA" from each segment as:
    doa_abs = (heading_relative_to_north + (180 - degree_relative_to_array)) % 360
This matches your example:
- If array heading is 90° (east) and degree_relative_to_array is 180 (along +direction),
  actual DOA = 90 + (180 - 180) = 90 (east).
- If degree_relative_to_array is 90 (broadside), actual DOA = 90 + (180 - 90) = 180 (south).

If you ever want the alternative (using slope sign), change `compute_actual_doa_azimuth` below.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math

import folium
from folium.plugins import MarkerCluster


# -------------------- small geo helpers --------------------

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _bearing_endpoint(lat: float, lon: float, az_deg: float, dist_m: float) -> Tuple[float, float]:
    """
    From (lat,lon), move dist_m meters at azimuth az_deg (clockwise from north).
    Uses simple local equirectangular meters-per-degree approximation (ok for short segments).
    """
    az = _deg2rad(az_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(_deg2rad(lat) + 1e-12)
    d_north = math.cos(az) * dist_m
    d_east  = math.sin(az) * dist_m
    dlat = d_north / m_per_deg_lat
    dlon = d_east  / m_per_deg_lon
    return lat + dlat, lon + dlon

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_source_track(path: Path) -> List[Tuple[float, float]]:
    """
    Returns a list of (lat, lon) points.
    Accepts:
      - GeoJSON LineString/MultiLineString (coords are [lon, lat, (alt)])
      - Plain JSON [[lat, lon], ...]
    """
    data = _load_json(path)
    # Plain list?
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
        # assume [lat, lon]
        return [(float(p[0]), float(p[1])) for p in data]

    # GeoJSON?
    if isinstance(data, dict) and "type" in data:
        if data["type"] == "FeatureCollection":
            feats = data.get("features", [])
            pts: List[Tuple[float, float]] = []
            for feat in feats:
                geom = feat.get("geometry", {})
                pts.extend(_extract_geojson_points(geom))
            return pts
        else:
            return _extract_geojson_points(data)  # geometry
    raise ValueError(f"Unrecognized source-track JSON format in {path}")

def _extract_geojson_points(geom: Dict[str, Any]) -> List[Tuple[float, float]]:
    t = geom.get("type")
    if t == "LineString":
        coords = geom.get("coordinates", [])
        return [(float(lat), float(lon)) for lon, lat, *rest in coords]
    if t == "MultiLineString":
        pts: List[Tuple[float, float]] = []
        for line in geom.get("coordinates", []):
            pts.extend((float(lat), float(lon)) for lon, lat, *rest in line)
        return pts
    return []

# -------------------- DOA composition --------------------

def compute_actual_doa_azimuth(heading_rel_north: float, degree_rel_array: float) -> float:
    """
    Convert (heading of array from lower→higher channel) + (degree_relative_to_array)
    into absolute DOA azimuth (deg from north, clockwise).

    Per your rule:
      - If degree_relative_to_array = 180 => DOA points along +array direction => azimuth = heading
      - If degree_relative_to_array = 90  => DOA is broadside to the right => azimuth = heading + 90
      - If degree_relative_to_array = 0   => DOA points along -array direction => azimuth = heading + 180

    This is: azimuth = heading + (180 - degree_relative_to_array)
    """
    return (heading_rel_north + (180.0 - degree_rel_array)) % 360.0

# -------------------- Folium plotting --------------------

def main():
    print("Plotting DOA map...")
    ap = argparse.ArgumentParser(description="Plot channels, source track, array segments and DOA rays on a Folium map.")
    ap.add_argument("--doa-json", required=True, type=Path, help="DOA JSON with segments and heading_relative_to_north.")
    ap.add_argument("--channel-positions", required=True, type=Path, help="channel_positions.json")
    ap.add_argument("--source-track", required=True, type=Path, help="Source track JSON (GeoJSON or [[lat,lon],...]).")
    ap.add_argument("--outfile", type=Path, default=Path("doa_map.html"), help="Output HTML map file.")
    ap.add_argument("--stride", type=int, default=1, help="Plot every Nth channel marker to reduce clutter.")
    ap.add_argument("--doa-ray-length-m", type=float, default=300.0, help="Length of DOA rays (meters).")
    ap.add_argument("--zoom-start", type=int, default=13, help="Initial zoom level.")
    args = ap.parse_args()

    # Load inputs
    doa = _load_json(args.doa_json)
    ch_pos_raw = _load_json(args.channel_positions)
    # normalize channel keys to int
    channels: Dict[int, List[float]] = {int(k): v for k, v in ch_pos_raw.items()}
    source_pts = _load_source_track(args.source_track)

    # Map center: average of channel positions (plotted subset)
    plot_chs = [i for i in sorted(channels.keys()) if i % args.stride == 0]
    lats = [channels[i][0] for i in plot_chs]
    lons = [channels[i][1] for i in plot_chs]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=args.zoom_start, control_scale=True)

    # ----- Layer: Channels -----
    layer_channels = folium.FeatureGroup(name="Channels", show=True).add_to(m)
    folium.PolyLine([(channels[i][0], channels[i][1]) for i in sorted(channels.keys())],
                    weight=2, opacity=0.5, tooltip="Channel path").add_to(layer_channels)

    cluster = MarkerCluster(name="Channel markers").add_to(layer_channels)
    for i in plot_chs:
        lat, lon, alt = channels[i]
        popup = folium.Popup(html=f"<b>Ch {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Alt: {alt:.1f} m", max_width=250)
        folium.CircleMarker([lat, lon], radius=3, weight=1, fill=True, fill_opacity=0.8,
                            popup=popup, tooltip=f"Ch {i}").add_to(cluster)

    # ----- Layer: Source track -----
    layer_source = folium.FeatureGroup(name="Source track", show=True).add_to(m)
    if source_pts:
        folium.PolyLine(source_pts, weight=3, color=None, tooltip="Source track").add_to(layer_source)

    # ----- Layers: Segments + DOA rays -----
    layer_segments = folium.FeatureGroup(name="Array segments", show=True).add_to(m)
    layer_doa = folium.FeatureGroup(name="DOA rays", show=True).add_to(m)

    # Go through each packet and subarray
    for packet, pdata in doa.items():
        for seg in pdata.get("subarrays", []):
            start_ch = int(seg["start_channel"])
            end_ch   = int(seg["end_channel"])
            # line between endpoints (from channel positions)
            if start_ch in channels and end_ch in channels:
                lat1, lon1, _ = channels[start_ch]
                lat2, lon2, _ = channels[end_ch]
                folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                                weight=4, opacity=0.7,
                                tooltip=f"Packet {packet}: ch {start_ch}-{end_ch} (len {seg.get('length')})").add_to(layer_segments)

            # DOA ray from center
            center = seg.get("center_position")
            heading = seg.get("heading_relative_to_north")
            beta = seg.get("degree_relative_to_array")

            if center and heading is not None and beta is not None:
                clat, clon = float(center[0]), float(center[1])
                doa_az = compute_actual_doa_azimuth(float(heading), float(beta))
                end_lat, end_lon = _bearing_endpoint(clat, clon, doa_az, args.doa_ray_length_m)

                folium.PolyLine([(clat, clon), (end_lat, end_lon)],
                                weight=3, opacity=0.9,
                                tooltip=f"Packet {packet}: DOA {doa_az:.1f}° (heading {heading:.1f}°, β {beta:.1f}°)").add_to(layer_doa)
                folium.CircleMarker([clat, clon], radius=4, color=None, fill=True, fill_opacity=1.0,
                                    tooltip=f"Center Packet {packet}").add_to(layer_doa)

    folium.LayerControl().add_to(m)
    m.save(str(args.outfile))
    print(f"Saved map to: {args.outfile}")

if __name__ == "__main__":
    main()
