from pathlib import Path
import json
import math
import folium

from dasprocessor.bearing_tools import (
    build_subarrays,
    get_cached_channel_gps_for_run,
    subarray_centers_and_headings,
)

def main():
    run_number = 2
    centers = [112, 208, 268, 552]
    aperture_len = 9

    subarrays = build_subarrays(centers, aperture_len, run_number)
    gps = get_cached_channel_gps_for_run(run_number)
    meta = subarray_centers_and_headings(subarrays, gps)

    # center the map on the mean of centers
    lat0 = sum(m["center_lat"] for m in meta.values()) / len(meta)
    lon0 = sum(m["center_lon"] for m in meta.values()) / len(meta)

    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

    # helper: draw a short arrow along heading
    def arrow_end(lat, lon, bearing_deg, length_m=200.0):
        th = math.radians(bearing_deg)
        dx = length_m * math.sin(th)
        dy = length_m * math.cos(th)
        lat2 = lat + (dy / 6378000.0) * (180.0 / math.pi)
        lon2 = lon + (dx / (6378000.0 * math.cos(math.radians(lat)))) * (180.0 / math.pi)
        return lat2, lon2

    for c in centers:
        lat = float(meta[c]["center_lat"])
        lon = float(meta[c]["center_lon"])
        hdg = float(meta[c]["heading_deg"])

        # point
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.9,
            popup=f"center={c}, heading={hdg:.2f}°",
        ).add_to(m)

        # heading arrow (simple line segment)
        lat2, lon2 = arrow_end(lat, lon, hdg, length_m=250.0)
        folium.PolyLine([[lat, lon], [lat2, lon2]], weight=3, opacity=0.8).add_to(m)

    out_path = Path(__file__).with_name("subarray_centers_run2.html")
    m.save(str(out_path))
    print(f"Map saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
