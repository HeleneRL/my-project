# channel = hydrophone
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np

# -----------------------

# Look into wether or not i can use Emils, to_cartesian here

# -----------------------

speed_of_sound = 1475.0  #  in m/s
sample_rate_hz = 25_000.0  # sample rate for timestamps


from dasprocessor.enu_geodetic import geodetic_to_ecef, ecef_to_enu_matrix, geodetic_to_enu



# -----------------------
# DOA helpers
# -----------------------

def solve_doa_plane_wave(positions_enu: np.ndarray,
                         timestamps: Dict[int, float],
                         channels: List[int],
                         speed_of_sound: float = speed_of_sound,
                         sample_rate_hz: float = sample_rate_hz,
                         reference_strategy: str = "first") -> Dict[str, Any]:
    """
    Estimate DOA unit vector u and az/el from ENU sensor positions and arrival times.

    positions_enu: (M,3) array of sensor positions in ENU (meters), ordered to match 'channels'
    timestamps:    dict {channel: t_arrival_seconds}
    channels:      list of channel IDs (order matches positions_enu rows)
    speed_of_sound: effective sound speed in m/s
    reference_strategy: "first" or "centroid_time" (pick reference for TDOAs)

    Returns:
        {
          'u_enu': [uE, uN, uU],
          'az_deg': float,   # azimuth clockwise from North, toward East
          'el_deg': float,   # elevation (up from horizontal)
          'used_channels': [...],
          'ref_channel': int,
          'method': 'WLS_norm',
          'residual_rms_s': float,  # fit RMS in seconds
        }

    """
    timestamps = {ch: ts / sample_rate_hz for ch, ts in timestamps.items()}  # convert to seconds

    # Build aligned time vector
    t = []
    for ch in channels:
        ch_str = str(ch)
        if ch_str not in timestamps:
            raise ValueError(f"Missing timestamp for channel {ch_str}")
        t.append(float(timestamps[ch_str]))
    t = np.array(t, dtype=float)

    M = positions_enu.shape[0]
    if M != len(channels) or M != len(t):
        raise ValueError("positions_enu, channels, and timestamps length mismatch.")

    if M < 4:
        raise ValueError("At least 4 sensors are needed to resolve 3D DOA under plane-wave TDOAs.")

    # Choose reference index for TDOAs
    if reference_strategy == "centroid_time":
        ref_idx = int(np.argmin(np.abs(t - np.mean(t))))
    else:
        ref_idx = 0  # first channel by default

    p0 = positions_enu[ref_idx]
    t0 = t[ref_idx]

    # Baselines and TDOAs vs reference
    b = positions_enu - p0  # (M,3)
    dt = t - t0             # (M,)

    # Remove the reference row
    mask = np.ones(M, dtype=bool)
    mask[ref_idx] = False
    A = b[mask, :]          # (M-1,3)
    d = speed_of_sound * dt[mask]  # (M-1,)

    # Weighted least squares (use identity weights unless you have per-channel timing variances)
    # u_ls = (A^T A)^-1 A^T d
    ATA = A.T @ A
    ATd = A.T @ d
    try:
        u_ls = np.linalg.solve(ATA, ATd)
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse if geometry is near-singular
        u_ls = np.linalg.pinv(ATA) @ ATd

    # Normalize to unit vector
    norm = np.linalg.norm(u_ls)
    if norm == 0:
        raise ValueError("Degenerate solution (||u||=0). Check geometry and timestamps.")
    u = u_ls / norm

    # Residual fit (in seconds)
    d_hat = A @ u
    residual = (d - d_hat) / speed_of_sound
    residual_rms_s = float(np.sqrt(np.mean(residual**2)))

    # Azimuth/elevation from ENU components
    uE, uN, uU = u.tolist()
    az = math.degrees(math.atan2(uE, uN))             # (-180, 180]
    el = math.degrees(math.asin(max(-1.0, min(1.0, uU))))  # clamp numeric noise

    return {
        "u_enu": [uE, uN, uU],
        "az_deg": az,
        "el_deg": el,
        "used_channels": channels,
        "ref_channel": channels[ref_idx],
        "method": "WLS_norm",
        "residual_rms_s": residual_rms_s,
        "c_mps": speed_of_sound,
    }

# -----------------------
# Main entry point
# -----------------------

def compute_doa_for_packet_groups(
    packet_groups: Dict[str, Dict[str, Dict[str, Any]]],
    channel_geo: Dict[int, List[float]],
    desired_packet: str,
    max_channel_number: int,
    speed_of_sound: float = 1475.0,
    reference_strategy: str = "first"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Add DOA estimates to each subset (startchannel-keyed) inside 'desired_packet'.

    Inputs:
      packet_groups:
          {
            packet_index: {
              startchannel: {
                "range": [start, end],
                "length": number_of_channels,
                "channels": [ch1, ch2, ...],  # detection order
                "timestamps": { ch: t_sec, ...}
              },
              ...
            },
            ...
          }

      channel_geo:
          { channel_number: [lat_deg, lon_deg, alt_m], ... }

      desired_packet: packet index to process
      max_channel_number: if subset's 'length' > this, only use the first 'max_channel_number'
      speed_of_sound: effective c in m/s (default 1500)
      reference_strategy: "first" or "centroid_time"

    Output:
      The same dict object (shallow-copied) but with, for each processed subset:
        entry["doa_relative_to_array"] = {
            'u_enu': [uE,uN,uU],
            'az_deg': float,
            'el_deg': float,
            'used_channels': [...],
            'ref_channel': int,
            'method': 'WLS_norm',
            'residual_rms_s': float,
            'c_mps': float,
            'enu_origin': {'lat_deg':..., 'lon_deg':..., 'alt_m':...}
        }
    """
    if desired_packet not in packet_groups:
        raise KeyError(f"desired_packet {desired_packet} not present in packet_groups.")

    # Make a shallow copy so we don't mutate the caller's dict unexpectedly
    out = dict(packet_groups)
    out_pkt = dict(out[desired_packet])

    for startchannel, subset in out_pkt.items():
        # Pull channels and apply max_channel_number rule
        channels_full: List[int] = list(subset.get("channels", []))
        n_full = int(subset.get("length", len(channels_full)))
        if n_full != len(channels_full):
            # trust 'channels' list more than 'length' if mismatch
            n_full = len(channels_full)

        if n_full == 0:
            subset["doa_relative_to_array"] = {
                "error": "No channels in subset."
            }
            continue

        if n_full > max_channel_number:
            channels = channels_full[:max_channel_number]
        else:
            channels = channels_full

        # Build geodetic list in same order as 'channels'
        try:
            geodetic_list = [channel_geo[ch] for ch in channels]
        except KeyError as e:
            subset["doa_relative_to_array"] = {
                "error": f"Missing channel position for channel {int(str(e))}."
            }
            continue

        # ENU origin: centroid of used channels (helps conditioning)
        lats = [geo[0] for geo in geodetic_list]
        lons = [geo[1] for geo in geodetic_list]
        alts = [geo[2] for geo in geodetic_list]
        ref_geo = (float(np.mean(lats)), float(np.mean(lons)), float(np.mean(alts)))

        # Convert to ENU
        try:
            positions_enu = geodetic_to_enu(geodetic_list, ref_geo)  # (M,3)
        except Exception as e:
            subset["doa_relative_to_array"] = {"error": f"Geodesy conversion failed: {e}"}
            continue

        


        # Solve DOA
        try:
            doa = solve_doa_plane_wave(
                positions_enu=positions_enu,
                timestamps=subset.get("timestamps", {}),
                channels=channels,
                speed_of_sound=speed_of_sound,
                reference_strategy=reference_strategy
            )
        except Exception as e:
            subset["doa_relative_to_array"] = {"error": str(e)}
            continue

        doa["enu_origin"] = {
            "lat_deg": ref_geo[0],
            "lon_deg": ref_geo[1],
            "alt_m": ref_geo[2],
        }

        # Attach result
        subset["doa_relative_to_array"] = doa

    # Put modified packet back
    out[desired_packet] = out_pkt
    return out



# for testing


#i want to import a json as a dict
import json
from dasprocessor.channel_gps import compute_channel_positions

with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\consecutive_peaks_v2.json", 'r') as file:
    consecutive_channel_dict = json.load(file)

channel_gps = compute_channel_positions(
        geojson_path=r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json",
        channel_count=1200,
        channel_distance=1.02,
        origin="start"  # or "end", "nearest", etc.
    )



def main() -> None:
    out = compute_doa_for_packet_groups(
        packet_groups=consecutive_channel_dict,
        channel_geo=channel_gps,
        desired_packet="69",
        max_channel_number=20,
        speed_of_sound=1475.0,
        reference_strategy="first"
    )
    print("DOA results for packet 69:")
    print(out["69"])

if __name__ == "__main__":
    main()  
    



# -----------------------
# Example (commented)
# -----------------------
# packet_groups = {
#   12: {
#     101: {
#       "range": [101, 120],
#       "length": 8,
#       "channels": [101,102,103,104,105,106,107,108],
#       "timestamps": {101:0.10023, 102:0.10075, 103:0.10111, 104:0.10142,
#                      105:0.10185, 106:0.10201, 107:0.10233, 108:0.10268}
#     }
#   }
# }
# channel_geo = {
#   101:[63.4305,10.3951,-100.0], 102:[63.4306,10.3953,-100.2], ... etc ...
# }
# out = compute_doa_for_packet_groups(packet_groups, channel_geo,
#                                     desired_packet=12, max_channel_number=6)
# print(out[12][101]["doa_relative_to_array"])
