import json
import os

# ----------------------------------------------------------------------
# INPUT / OUTPUT PATHS
# ----------------------------------------------------------------------
in_path = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_reoriented.geojson"

root, ext = os.path.splitext(in_path)
out_path = root + "_sampled" + ext

# sampling step (keep every nth point)
STEP = 5

# ----------------------------------------------------------------------
# LOAD GEOJSON
# ----------------------------------------------------------------------
with open(in_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

features = gj.get("features", [])

# ----------------------------------------------------------------------
# RESAMPLE EACH MULTILINESTRING
# ----------------------------------------------------------------------
for feat in features:
    geom = feat.get("geometry", {})
    if geom.get("type") != "MultiLineString":
        continue

    new_lines = []
    for line in geom.get("coordinates", []):
        if not line:
            new_lines.append(line)
            continue

        # take every STEP-th point, always keep the last one
        sampled = line[0::STEP]
        if sampled[-1] != line[-1]:
            sampled.append(line[-1])

        new_lines.append(sampled)

    geom["coordinates"] = new_lines

    # (optional) keep Shape__Len as is, since physical length hasn't changed
    # If you want, you could recompute it here.

# ----------------------------------------------------------------------
# SAVE NEW GEOJSON
# ----------------------------------------------------------------------
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(gj, f, indent=2)

print("Wrote sampled GeoJSON to:")
print(" ", out_path)
