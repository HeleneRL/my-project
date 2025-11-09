#libs\dasprocessor\scripts\doa_relative_array.py
"""
Compute direction of arrival (relative to array) for the top-M consecutive channel runs.

Input JSON format (typical "both_ts" produced earlier):
{
  "packet_id": {
    "start_channel_as_str": {
      "range": [s, e],
      "length": N,
      "channels": [ ... ],
      "timestamps": { "ch": ts_in_samples, ... }
    },
    ...
  }
}

Output JSON format:
{
  "packet_id": {
    "subarray_count": M,
    "subarrays": [
      {
        "start_channel": s,
        "end_channel": e,
        "length": N,
        "degree_relative_to_array": beta_deg,
        "slope_s_per_channel": b,
        "residual_rms_s": rms,
        "used_points": K
      },
      ...
    ]
  },
  ...
}
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, Any, List, Tuple
import importlib
import sys


# ------------------------- utilities -------------------------




def load_constants_module(constants_path: Path):
    pkg_dir = constants_path.parent           # .../dasprocessor
    pkg_name = pkg_dir.name                   # "dasprocessor"
    pkg_parent = str(pkg_dir.parent)          # path to folder containing the package

    # Ensure it's a real package (has __init__.py)
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        raise RuntimeError(f"{pkg_dir} is not a package (missing __init__.py). Create it (can be empty).")

    # Temporarily put the parent on sys.path and import package.module
    sys.path.insert(0, pkg_parent)
    try:
        return importlib.import_module(f"{pkg_name}.constants")
    finally:
        sys.path.pop(0)




def linear_fit_slope(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    OLS slope b for y = a + b x; returns (b, a, rss).
    """
    n = len(x)
    xm = sum(x) / n
    ym = sum(y) / n
    sxx = sum((xi - xm) ** 2 for xi in x)
    if sxx == 0:
        return 0.0, ym, 0.0
    b = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / sxx
    a = ym - b * xm
    rss = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    return b, a, rss


def mad(values: List[float]) -> float:
    """
    Median Absolute Deviation (scale-agnostic; no 1.4826 factor since we just
    use it for robust thresholding).
    """
    m = median(values)
    return median([abs(v - m) for v in values])


def robust_line_fit(x: List[float], y: List[float], max_passes: int = 2) -> Tuple[float, float, List[int], float]:
    """
    Simple robust fit: OLS -> remove |residual| > 4*MAD -> OLS again.
    Returns (slope b, intercept a, kept_indices, residual_rms).
    """
    idx = list(range(len(x)))
    for _ in range(max_passes):
        xx = [x[i] for i in idx]
        yy = [y[i] for i in idx]
        if len(xx) < 2:
            break
        b, a, rss = linear_fit_slope(xx, yy)
        res = [yy_i - (a + b * xx_i) for xx_i, yy_i in zip(xx, yy)]
        m = mad(res)
        # numeric floor so we don't nuke good fits that are super-flat
        thr = max(5e-7, 4.0 * m)  # seconds
        keep = [j for j, r in zip(idx, res) if abs(r) <= thr]
        if len(keep) == len(idx):  # stable
            rms = math.sqrt(rss / len(xx)) if len(xx) > 0 else float("nan")
            return b, a, idx, rms
        idx = keep
    # final fit
    xx = [x[i] for i in idx]
    yy = [y[i] for i in idx]
    if len(xx) >= 2:
        b, a, rss = linear_fit_slope(xx, yy)
        rms = math.sqrt(rss / len(xx)) if len(xx) > 0 else float("nan")
    else:
        b, a, rms = 0.0, (yy[0] if yy else 0.0), float("nan")
    return b, a, idx, rms


def slope_to_beta_deg(b_s_per_ch: float, c: float, d: float) -> float:
    """
    Convert slope b (seconds per channel) into angle beta (deg) relative to array axis.
    b = (d/c) * cos(beta)  =>  beta = arccos( clamp(c*b/d, -1, 1) ).
    """
    x = (c * b_s_per_ch) / d
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))


# ------------------------- main processing -------------------------

def process_packet_runs(
    packet: str,
    runs: Dict[str, Any],
    sample_rate: float,
    channel_distance: float,
    c: float,
    min_points: int = 3,
) -> Dict[str, Any]:
    """
    For one packet: compute DOA for each run.
    """
    subarrays = []

    # Keep the input ordering by run start (keys are start-channel strings).
    for start_str, info in sorted(runs.items(), key=lambda kv: int(kv[0])):
        chs = info.get("channels") or []
        ts_map = info.get("timestamps") or {}
        if not chs or not ts_map:
            continue

        chs = sorted(int(ch) for ch in chs)
        # Align timestamps in seconds
        try:
            t_sec = [float(ts_map[str(ch)]) / sample_rate for ch in chs if str(ch) in ts_map]
        except Exception as e:
            # If any parsing fails, skip this run
            continue
        if len(t_sec) != len(chs):
            # Skip incomplete runs
            continue

        # Robust fit
        if len(chs) < min_points:
            # too short, but still provide a flat estimate
            b = 0.0
            kept_idx = list(range(len(chs)))
            rms = float("nan")
        else:
            b, a, kept_idx, rms = robust_line_fit(chs, t_sec)

        kept_ch = [chs[i] for i in kept_idx]
        if len(kept_ch) < min_points:
            # not enough good points left
            continue

        # Degree relative to array
        beta = slope_to_beta_deg(b, c=c, d=channel_distance)

        out = {
            "start_channel": int(info["range"][0]) if "range" in info else int(kept_ch[0]),
            "end_channel": int(info["range"][1]) if "range" in info else int(kept_ch[-1]),
            "length": int(info.get("length", len(kept_ch))),
            "degree_relative_to_array": float(beta),
            "slope_s_per_channel": float(b),
            "residual_rms_s": float(rms),
            "used_points": int(len(kept_ch)),
        }
        subarrays.append(out)

    return {
        "subarray_count": len(subarrays),
        "subarrays": subarrays,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute DOA (relative to array) from top-M consecutive channel runs.")
    ap.add_argument("runs_json", type=Path, help="Input JSON with per-packet top-M runs (must include timestamps).")
    ap.add_argument("constants_py", type=Path, help="Path to libs\\dasprocessor\\constants.py")
    ap.add_argument("--date", required=True, help='Key into constants.properties, e.g. "2024-05-03"')
    ap.add_argument("--run", type=int, required=True, help="Run index inside constants.properties[date], e.g. 2")
    ap.add_argument("--output", type=Path, default=Path("packet_doa.json"), help="Output JSON path")
    ap.add_argument("--c", type=float, default=1475.0, help="Wave speed (m/s). Default 1475")
    ap.add_argument("--min-points", type=int, default=3, help="Minimum inlier points required to keep a run")
    ap.add_argument("--indent", type=int, default=2, help="Indent for pretty JSON")
    args = ap.parse_args()

    # Load constants module and pick properties
    consts = load_constants_module(args.constants_py)
    try:
        props = consts.properties[args.date][args.run]
    except Exception as e:
        raise RuntimeError(f"Could not find properties['{args.date}'][{args.run}] in {args.constants_py}") from e

    sample_rate = float(props["sample_rate"])
    channel_distance = float(props["channel_distance"])

    # Read runs JSON
    with args.runs_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Process packets
    results: Dict[str, Any] = {}
    for packet, runs in data.items():
        results[packet] = process_packet_runs(
            packet=packet,
            runs=runs,
            sample_rate=sample_rate,
            channel_distance=channel_distance,
            c=args.c,
            min_points=args.min_points,
        )

    # Write output
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote DOA results to {args.output}")


if __name__ == "__main__":
    main()
