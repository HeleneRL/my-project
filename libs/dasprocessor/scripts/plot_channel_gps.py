#libs\dasprocessor\scripts\plot_channel_gps.py


from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import folium
from folium.plugins import MarkerCluster

# Import your function as a package module
from dasprocessor.channel_gps import compute_channel_positions
from dasprocessor.constants import get_run   

def parse_latlon(text: str) -> Tuple[float, float]:
    """
    Parse "lat,lon" or "lat lon" into (lat, lon) floats.
    """
    s = text.replace(",", " ").split()
    if len(s) != 2:
        raise argparse.ArgumentTypeError("Expected 'lat,lon' or 'lat lon'")
    lat = float(s[0])
    lon = float(s[1])
    return lat, lon

def load_cable_polyline(geojson_path: Path) -> List[Tuple[float, float, float]]:
    """
    Returns [(lat, lon, alt), ...] concatenated from the MultiLineString.
    """
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    feats = gj.get("features", [])
    if not feats:
        return []

    geom = feats[0].get("geometry", {})
    if geom.get("type") != "MultiLineString":
        return []

    coords_multi = geom.get("coordinates", [])
    pts: List[Tuple[float, float, float]] = []
    for line in coords_multi:
        for c in line:
            if len(c) >= 3:
                lon, lat, alt = float(c[0]), float(c[1]), float(c[2])
            else:
                lon, lat, alt = float(c[0]), float(c[1]), 0.0
            pts.append((lat, lon, alt))
    return pts

def main():
    ap = argparse.ArgumentParser(description="Compute and plot channel GPS positions on a folium map.")
    ap.add_argument("--geojson", type=Path, default=r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json", help="Path to cable-layout.json (GeoJSON MultiLineString).")
    ap.add_argument("--channel-count", type=int, default=1200, help="Total number of channels (e.g. 1200).")
    ap.add_argument("--channel-distance",  type=float, default=1.02, help="Channel spacing in meters (e.g. 1.02).")

    ap.add_argument("--origin", choices=["start", "end", "nearest", "chainage"], default="start",
                    help="Where channel 0 sits along the cable (default: start).")
    ap.add_argument("--nearest-to", type=parse_latlon, default=None,
                    help="lat,lon used if --origin nearest (e.g. \"63.44650,10.35220\").")
    ap.add_argument("--origin-offset-m", type=float, default=0.0,
                    help="Extra along-track offset (meters) after origin selection (default: 0).")

    ap.add_argument("--stride", type=int, default=1,
                    help="Plot every Nth channel to reduce clutter (default: 1 = all).")
    ap.add_argument("--marker-cluster", action="store_true",
                    help="Use a marker cluster instead of individual markers.")
    ap.add_argument("--show-cable", action="store_true",
                    help="Also draw the cable polyline from the GeoJSON.")
    ap.add_argument("--outfile", type=Path, default=Path("channel_map.html"),
                    help="Output HTML file path (default: channel_map.html).")

    args = ap.parse_args()

    # Compute channel positions
    channels: Dict[int, List[float]] = compute_channel_positions(
        geojson_path=args.geojson,
        channel_count=args.channel_count,
        channel_distance=args.channel_distance,
        origin=args.origin,
        origin_offset_m=args.origin_offset_m,
        nearest_to=args.nearest_to,
        interpolate_missing_alts=True,
    )

    # Map center = mean of all plotted points (or cable mid if available)
    plot_indices = [i for i in sorted(channels.keys()) if (i % args.stride == 0)]
    if not plot_indices:
        raise SystemExit("No channels to plot (check --stride).")

    lats = [channels[i][0] for i in plot_indices]
    lons = [channels[i][1] for i in plot_indices]
    alts = [channels[i][2] for i in plot_indices]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Optionally draw the cable polyline
    if args.show_cable:
        cable_pts = load_cable_polyline(args.geojson)
        if cable_pts:
            folium.PolyLine([(lat, lon) for (lat, lon, _alt) in cable_pts],
                            weight=3, opacity=0.6, tooltip="Cable polyline").add_to(m)

    # Add channel markers
    if args.marker_cluster:
        cluster = MarkerCluster(name="Channels").add_to(m)
        marker_parent = cluster
    else:
        marker_parent = m

    # Also draw a thin polyline through channel sequence (subsampled by stride)
    folium.PolyLine([(channels[i][0], channels[i][1]) for i in plot_indices],
                    weight=2, opacity=0.5, tooltip="Channel path").add_to(m)

    for i in plot_indices:
        lat, lon, alt = channels[i]
        popup = folium.Popup(
            html=f"<b>Channel {i}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Alt: {alt:.1f} m",
            max_width=250
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            weight=1,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
            tooltip=f"Ch {i} (alt {alt:.1f} m)"
        ).add_to(marker_parent)

    folium.LayerControl().add_to(m)
    m.save(str(args.outfile))
    print(f"Saved map to: {args.outfile}")

if __name__ == "__main__":
    main()
