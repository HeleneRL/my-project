import json
import numpy as np
from pymap3d import geodetic2enu, enu2geodetic

# ----------------------------------------------------------------------
# FILE PATHS
# ----------------------------------------------------------------------
cable_in = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"        # your big dict
cable_out = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_adjusted.json"       # adjusted JSON dict
geojson_out = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_adjusted.geojson"  # new GeoJSON file

# ----------------------------------------------------------------------
# LOAD ORIGINAL LAYOUT
# ----------------------------------------------------------------------
with open(cable_in, "r") as f:
    cable = json.load(f)  # dict: "chan" -> [lat, lon, depth]

# Sort by channel index
channels = sorted(cable.keys(), key=lambda k: int(k))
n = len(channels)

lats = np.array([cable[ch][0] for ch in channels])
lons = np.array([cable[ch][1] for ch in channels])
deps = np.array([cable[ch][2] for ch in channels])  # keep depths as-is

# ----------------------------------------------------------------------
# CONVERT TO LOCAL ENU (x,y)
# ----------------------------------------------------------------------
# Use channel 0 as ENU origin
lat0 = lats[0]
lon0 = lons[0]
alt0 = 0.0

xs = np.zeros(n)
ys = np.zeros(n)

for i in range(n):
    x, y, z = geodetic2enu(lats[i], lons[i], 0.0, lat0, lon0, alt0)
    xs[i] = x
    ys[i] = y

# ----------------------------------------------------------------------
# ORIGINAL SEGMENT LENGTHS (KEEP THESE)
# ----------------------------------------------------------------------
dx = np.diff(xs)
dy = np.diff(ys)
ds = np.sqrt(dx**2 + dy**2)  # length between channel i and i+1 (horizontal)

total_length = float(np.sum(ds))  # in meters, used for Shape__Len

# ----------------------------------------------------------------------
# HELPER: DESIRED ORIENTATION (DEG FROM EAST, CCW) PER CHANNEL
# ----------------------------------------------------------------------
def desired_theta_deg(ch):
    """
    Piecewise orientation schedule based on DAS 'story'.
    ch is the channel index (int).
    Returns an angle in degrees from East, CCW.
    Adjust the numbers as you like.
    """
    # 0–80: keep original orientation (use local slope)
    if ch < 80:
        i = ch
        if i <= 0:
            i = 1
        if i >= n - 1:
            i = n - 2
        dx_local = xs[i+1] - xs[i-1]
        dy_local = ys[i+1] - ys[i-1]
        return np.degrees(np.arctan2(dy_local, dx_local))

    # 80–90: straighten to East (0°)
    if 80 <= ch <= 90:
        # orientation at 80 based on original geometry
        i0 = 80
        if i0 <= 0:
            i0 = 1
        if i0 >= n - 1:
            i0 = n - 2
        dx_local = xs[i0+1] - xs[i0-1]
        dy_local = ys[i0+1] - ys[i0-1]
        theta80 = np.degrees(np.arctan2(dy_local, dx_local))
        t = (ch - 80) / 10.0
        return (1 - t) * theta80 + t * 0.0

    # 90–130: bend into NE ~50°
    if 90 < ch <= 130:
        # interpolate from 0° at 90 to 50° at 130
        t = (ch - 90) / 40.0
        return (1 - t) * 0.0 + t * 50.0

    # 130–250: stable NE ~50°
    if 130 < ch <= 250:
        return 50.0

    # 250–300: slightly relax to ~40°
    if 250 < ch <= 300:
        # interpolate from 50° at 250 to 40° at 300
        t = (ch - 250) / 50.0
        return (1 - t) * 50.0 + t * 40.0

    # Beyond 300: keep original orientation
    i = ch
    if i <= 0:
        i = 1
    if i >= n - 1:
        i = n - 2
    dx_local = xs[i+1] - xs[i-1]
    dy_local = ys[i+1] - ys[i-1]
    return np.degrees(np.arctan2(dy_local, dx_local))

# ----------------------------------------------------------------------
# RECONSTRUCT NEW ENU PATH WITH DESIRED ORIENTATION
# ----------------------------------------------------------------------
x_new = np.zeros(n)
y_new = np.zeros(n)

# Start at original channel 0 location in ENU
x_new[0] = xs[0]
y_new[0] = ys[0]

for i in range(1, n):
    ch = int(channels[i])  # actual channel index
    theta_deg = desired_theta_deg(ch)
    theta_rad = np.radians(theta_deg)

    step = ds[i-1]  # preserve original segment length

    x_new[i] = x_new[i-1] + step * np.cos(theta_rad)
    y_new[i] = y_new[i-1] + step * np.sin(theta_rad)

# ----------------------------------------------------------------------
# CONVERT BACK TO LAT/LON, BUILD NEW DICT
# ----------------------------------------------------------------------
new_cable = {}

for i, ch in enumerate(channels):
    lat, lon, alt = enu2geodetic(x_new[i], y_new[i], 0.0, lat0, lon0, alt0)
    depth = float(deps[i])  # keep original depth
    new_cable[ch] = [float(lat), float(lon), depth]

# ----------------------------------------------------------------------
# SAVE ADJUSTED JSON (DICT FORMAT)
# ----------------------------------------------------------------------
with open(cable_out, "w") as f:
    json.dump(new_cable, f, indent=2)

print("Saved adjusted cable layout dict to:")
print("  ", cable_out)

# ----------------------------------------------------------------------
# BUILD GEOJSON FeatureCollection WITH MultiLineString
# ----------------------------------------------------------------------
# Coordinates in GeoJSON: [lon, lat, depth]

coords_line = []
for ch in channels:
    lat, lon, depth = new_cable[ch]
    coords_line.append([lon, lat, depth])

feature = {
    "type": "Feature",
    "properties": {
        "fid_1": 1.0,
        "Label": "SDP Tether",
        "Location": "On Seabed",
        "Shape__Len": total_length,  # horizontal length in meters
    },
    "geometry": {
        "type": "MultiLineString",
        # one line made of all points; if you want multiple segments,
        # you can split coords_line into sublists.
        "coordinates": [coords_line],
    },
}

geojson = {
    "type": "FeatureCollection",
    "name": "TBS_Fjordlab_Tether_Adjusted",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84",
        },
    },
    "features": [feature],
}

with open(geojson_out, "w") as f:
    json.dump(geojson, f, indent=2)

print("Saved adjusted cable GeoJSON to:")
print("  ", geojson_out)
