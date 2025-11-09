#libs\dasprocessor\scripts\consecutive_hits.py

import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Iterable, Any

def find_consecutive_runs(nums: Iterable[int]) -> List[Tuple[int, int, List[int]]]:
    """Return (start, end, full_list) for each maximal consecutive run."""
    runs: List[Tuple[int, int, List[int]]] = []
    nums = sorted(set(nums))
    if not nums:
        return runs
    start = prev = nums[0]
    acc = [start]
    for x in nums[1:]:
        if x == prev + 1:
            acc.append(x)
        else:
            runs.append((start, prev, acc[:]))
            start = x
            acc = [x]
        prev = x
    runs.append((start, prev, acc))
    return runs

def _apply_top_m(runs_dict: Dict[str, Any], top_m: int | None) -> Dict[str, Any]:
    if top_m is None:
        return runs_dict
    # Sort by length descending, then by start channel ascending
    items = []
    for k, v in runs_dict.items():
        if isinstance(v, dict) and "length" in v:
            length = v["length"]
        elif isinstance(v, list) and len(v) == 2:
            length = v[1] - v[0] + 1
        else:
            # fallback
            length = 0
        items.append((int(k), v, int(length)))
    items.sort(key=lambda t: (-t[2], t[0]))
    # Keep only top M, restore ordering by start channel
    kept = sorted(items[:top_m], key=lambda t: t[0])
    return {str(k): v for k, v, _ in kept}

def build_output_for_packet(
    ch2ts: Dict[str, int],
    mode: str = "ranges",
    top_m: int | None = None
) -> Dict[str, object]:
    """
    ch2ts: mapping 'channel' -> timestamp (strings ok)
    mode:
      - "ranges":  {start: [start, end]}
      - "lists":   {start: [list_of_channels]}
      - "both":    {start: {"range":[s,e], "length":n, "channels":[...]}}
      - "both_ts": {start: {"range":[s,e], "length":n, "channels":[...], "timestamps": {ch: ts, ...}}}
    """
    # parse numeric channels, keep only numeric keys
    ints = [int(c) for c in ch2ts.keys() if str(c).isdigit()]
    result: Dict[str, object] = {}

    for s, e, lst in find_consecutive_runs(ints):
        key = str(s)
        if mode == "ranges":
            result[key] = [s, e]
        elif mode == "lists":
            result[key] = lst
        elif mode == "both":
            result[key] = {"range": [s, e], "length": len(lst), "channels": lst}
        elif mode == "both_ts":
            timestamps = {str(ch): ch2ts[str(ch)] for ch in lst if str(ch) in ch2ts} 
            result[key] = {
                "range": [s, e],
                "length": len(lst),
                "channels": lst,
                "timestamps": timestamps
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # deterministic order by start channel
    ordered = {k: result[k] for k in sorted(result.keys(), key=lambda z: int(z))}
    # optionally keep only the M longest runs
    ordered = _apply_top_m(ordered, top_m)
    return ordered

def transform_runs(
    data: Dict[str, Dict[str, int]],
    mode: str = "ranges",
    top_m: int | None = None
) -> Dict[str, Dict[str, object]]:
    """data: {packet: {channel: timestamp}} -> {packet: runs_by_start}"""
    out: Dict[str, Dict[str, object]] = {}
    for packet, ch_map in data.items():
        out[packet] = build_output_for_packet(ch_map, mode=mode, top_m=top_m)
    # sort packets numerically
    return {k: out[k] for k in sorted(out.keys(), key=lambda z: int(z) if str(z).isdigit() else z)}

def main():
    print("Starting consecutive channel runs extractor")
    ap = argparse.ArgumentParser(description="Collapse per-packet channels into consecutive runs (with optional timestamps).")
    ap.add_argument("input", type=Path, help="Input JSON: {packet: {channel: timestamp}}")
    ap.add_argument("output", type=Path, help="Output JSON")
    ap.add_argument("--mode", choices=["ranges", "lists", "both", "both_ts"], default="both_ts",
                    help="Output format per run (default: both_ts)")
    ap.add_argument("--top-m", type=int, default=None,
                    help="Keep only the M longest runs per packet")
    ap.add_argument("--indent", type=int, default=2, help="Pretty-print indent (default: 2)")
    args = ap.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = transform_runs(data, mode=args.mode, top_m=args.top_m)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote consecutive-run summary to {args.output}")

if __name__ == "__main__":
    main()
