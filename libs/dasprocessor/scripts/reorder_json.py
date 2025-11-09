#libs\dasprocessor\scripts\reorder_json.py

import json
from pathlib import Path
import argparse
from collections import defaultdict

def flip_channel_packet_timestamp(data: dict) -> dict:
    """
    Convert {channel: {packet: timestamp}} to {packet: {channel: timestamp}}.
    Keeps keys as strings (JSON style) and values as-is.
    """
    by_packet = defaultdict(dict)

    for channel_str, packets in data.items():
        if not isinstance(packets, dict):
            raise ValueError(f"Expected dict for channel '{channel_str}', got {type(packets).__name__}")
        for packet_str, ts in packets.items():
            # If the same (packet, channel) shows up twice with conflicting timestamps, warn/override.
            if channel_str in by_packet[packet_str] and by_packet[packet_str][channel_str] != ts:
                # Last one wins, but you could also raise here if you prefer strictness.
                pass
            by_packet[packet_str][channel_str] = ts

    # Convert back to regular dict for JSON dumping
    return {packet: dict(ch_map) for packet, ch_map in by_packet.items()}


def main():
    parser = argparse.ArgumentParser(description="Flip JSON from {channel:{packet:timestamp}} to {packet:{channel:timestamp}}.")
    parser.add_argument("input", type=Path, help="Path to input JSON file")
    parser.add_argument("output", type=Path, help="Path to output JSON file")
    parser.add_argument("--indent", type=int, default=2, help="Indent level for pretty JSON (default: 2)")
    parser.add_argument("--sort-keys", action="store_true", help="Sort keys in the output JSON")
    args = parser.parse_args()

    # Read
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Transform
    flipped = flip_channel_packet_timestamp(data)

    # (Optional) sort keys for readability
    if args.sort_keys:
        flipped = {k: dict(sorted(v.items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else kv[0])))
                   for k, v in sorted(flipped.items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else kv[0]))}

    # Write
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(flipped, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote flipped structure to: {args.output}")

if __name__ == "__main__":
    main()
