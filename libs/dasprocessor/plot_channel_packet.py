# libs/dasprocessor/plot_channel_packet.py
from __future__ import annotations

import math
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from dasprocessor.constants import get_run, frequency_bands
from dasprocessor.saveandload import load_interrogator_data


# --------------------------------------------------------------------
# CONFIG DEFAULTS
# --------------------------------------------------------------------

DATE_STR = "2024-05-03"
RUN_NUMBER_DEFAULT = 2

# Where the interrogator HDF5 files live
DEFAULT_BASENAME = r"D:\DASComms_25kHz_GL_2m\20240503\dphi"

# Where cached NPZs are stored
DEFAULT_CACHEPATH = r"D:\backups"

# Default signal band (for filtered view)
DEFAULT_SIGNAL_KIND = "B_4"    # uses frequency_bands["B_4"]

# Default time window length around packet center [seconds]
DEFAULT_WINDOW_S = 2.0

# Step in seconds between DAS blocks used when caching
DEFAULT_STEP_S = 10

# How many channels per cached block (e.g. 112–124 ⇒ width = 12)
CHANNEL_BLOCK_WIDTH = 12


# --------------------------------------------------------------------
# TIME HELPERS
# --------------------------------------------------------------------

def hms_to_seconds(h: int, m: int, s: float) -> float:
    """Convert (h, m, s) to seconds since midnight."""
    return 3600.0 * h + 60.0 * m + float(s)


# --------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------

def compute_packet_center_time_s(date: str, run: int, packet_idx: int) -> float:
    """
    Compute absolute time-of-day (seconds since midnight) of the *center*
    of JANUS packet 'packet_idx' for the given date and run, using constants.py.
    """
    run_props = get_run(date, run)
    fs = run_props["sample_rate"]
    seq_start_h, seq_start_m, seq_start_s = run_props["sequence_start"]
    seq_period_samples = run_props["sequence_period"]
    seq_period_s = seq_period_samples / fs

    t0 = hms_to_seconds(seq_start_h, seq_start_m, seq_start_s)
    # Approximate the "center" of each packet by sequence_start + k * period
    t_packet_center = t0 + packet_idx * seq_period_s
    return t_packet_center


def load_raw_and_filtered_window(
    basename: str | Path,
    cachepath: str | Path,
    date: str,
    run: int,
    channel: int,
    packet_idx: int,
    window_s: float,
    signal_kind: str,
    step_s: int = DEFAULT_STEP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load raw and filtered DAS data for a *whole run* (for a block of channels),
    then extract a time window around the given packet & channel.

    Returns:
        t_rel          : time axis [s], centered at packet (0 = packet center)
        y_raw_win      : raw strain samples in window
        y_filt_win     : filtered strain samples in window
        y_filt_full    : filtered strain for the full run (for spectrogram)
        fs             : sampling frequency
    """
    run_props = get_run(date, run)
    fs = run_props["sample_rate"]

    # --- 1) Run-level time range from constants.py ---
    (run_start_h, run_start_m, run_start_s), (run_stop_h, run_stop_m, run_stop_s) = run_props["time_range"]

    start_hms = (run_start_h, run_start_m, run_start_s)
    stop_hms  = (run_stop_h,  run_stop_m,  run_stop_s)

    # This is the time-of-day at sample index 0 in the loaded arrays
    t_run_start_s = hms_to_seconds(*start_hms)

    print(f"Run {run} time range: {start_hms} -> {stop_hms}")

    # --- 2) Channel *block* slice for loader ---
    block_start = channel                  # we assume cache was created with this as start
    block_stop = channel + CHANNEL_BLOCK_WIDTH
    ch_slice = slice(block_start, block_stop)  # e.g. 112–124

    print(f"Channel block slice: [{block_start}:{block_stop}), step = {step_s} s")

    # Index of our requested channel within this block:
    channel_index = channel - block_start  # should be 0 with this choice

    # --- 3) Load RAW data for the whole run (from cache or HDF5) ---
    raw = load_interrogator_data(
        basename=basename,
        start=start_hms,
        stop=stop_hms,
        step=step_s,
        channels=ch_slice,
        out="npz",
        on_fnf="cache",
        filter_band=None,     # raw
        verbose=False,
        cachepath=str(cachepath),
    )
    y_raw_block = raw["y"]  # shape (Nsamples, Nblock)
    fs_loaded = raw["fs"]
    assert abs(fs_loaded - fs) < 1e-3, "Sample rate mismatch?"
    n_samples = y_raw_block.shape[0]

    # Pick our channel from the block
    y_raw_full = y_raw_block[:, channel_index]

    # --- 4) Load FILTERED data (same interval, same channel block) ---
    fmin, fmax = frequency_bands[signal_kind]
    filt = load_interrogator_data(
        basename=basename,
        start=start_hms,
        stop=stop_hms,
        step=step_s,
        channels=ch_slice,
        out="npz",
        on_fnf="cache",
        filter_band=(fmin, fmax),
        verbose=False,
        cachepath=str(cachepath),
    )
    y_filt_block = filt["y"]
    y_filt_full = y_filt_block[:, channel_index]

    # --- 5) Compute packet center in samples relative to run start ---
    t_packet_center = compute_packet_center_time_s(date, run, packet_idx)
    dt_packet_from_run_start = t_packet_center - t_run_start_s
    idx_center = int(round(dt_packet_from_run_start * fs))

    print(f"Packet {packet_idx}:")
    print(f"  center time ≈ {t_packet_center:.3f} s since midnight")
    print(f"  run start time = {t_run_start_s:.3f} s")
    print(f"  -> center sample index ≈ {idx_center} (of {n_samples})")

    # --- 6) Extract time window around packet center ---
    half_win_samples = int(round(window_s * fs / 2.0))
    i0 = max(0, idx_center - half_win_samples)
    i1 = min(n_samples, idx_center + half_win_samples)

    sample_indices = np.arange(i0, i1)
    t_abs = t_run_start_s + sample_indices / fs
    t_rel = t_abs - t_packet_center  # 0 at packet center

    y_raw_win = y_raw_full[i0:i1]
    y_filt_win = y_filt_full[i0:i1]

    print(f"  fs = {fs:.1f} Hz, window samples [{i0}:{i1}] "
          f"({(i1 - i0) / fs:.3f} s)")

    return t_rel, y_raw_win, y_filt_win, y_filt_full, fs


def plot_all(
    t_rel: np.ndarray,
    y_raw_win: np.ndarray,
    y_filt_win: np.ndarray,
    y_filt_full: np.ndarray,
    fs: float,
    channel: int,
    packet_idx: int,
    signal_kind: str,
) -> None:
    """
    Make three figures:
      1) Raw time series (window around packet)
      2) Filtered time series (window)
      3) Full-run spectrogram of filtered data
    """
    fmin, fmax = frequency_bands[signal_kind]

    # --- 1) Raw time series (window) ---
    plt.figure(figsize=(10, 4))
    plt.plot(t_rel, y_raw_win)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.title(f"Channel {channel}, packet {packet_idx} – RAW")
    plt.xlabel("Time relative to packet center [s]")
    plt.ylabel("Strain (raw, arb.)")
    plt.grid(True, alpha=0.3)

    # --- 2) Filtered time series (window) ---
    plt.figure(figsize=(10, 4))
    plt.plot(t_rel, y_filt_win)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.title(
        f"Channel {channel}, packet {packet_idx} – Filtered {signal_kind} "
        f"({fmin:.0f}-{fmax:.0f} Hz)"
    )
    plt.xlabel("Time relative to packet center [s]")
    plt.ylabel("Strain (filtered, arb.)")
    plt.grid(True, alpha=0.3)

    # --- 3) Full-run spectrogram of filtered data ---
    plt.figure(figsize=(12, 6))

    # Spectrogram of the entire run for this channel (filtered)
    Pxx, freqs, bins, im = plt.specgram(
        y_filt_full,
        NFFT=2048,
        Fs=fs,
        noverlap=1024,
        cmap="viridis",
    )
    # bins are in seconds from start of this signal
    plt.ylim(0, min(4000, fs / 2))
    plt.xlabel("Time from run start [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(
        f"Channel {channel}, full run spectrogram (filtered {signal_kind}, "
        f"{fmin:.0f}-{fmax:.0f} Hz)"
    )
    plt.colorbar(label="Power (dB-ish)")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DAS strain & spectrogram for a channel/packet."
    )
    parser.add_argument("--channel", type=int, required=True, help="Channel index (0-based)")
    parser.add_argument("--packet", type=int, required=True, help="Packet index (0-based)")
    parser.add_argument("--run", type=int, default=RUN_NUMBER_DEFAULT, help="Run number (default 2)")
    parser.add_argument("--date", type=str, default=DATE_STR, help="Date string (default 2024-05-03)")
    parser.add_argument("--basename", type=str, default=DEFAULT_BASENAME, help="Interrogator data basename")
    parser.add_argument("--cachepath", type=str, default=DEFAULT_CACHEPATH, help="Cache directory for NPZ/MAT")
    parser.add_argument("--signal-kind", type=str, default=DEFAULT_SIGNAL_KIND,
                        choices=frequency_bands.keys(),
                        help="Signal band name (for filtered view)")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                        help="Time window [s] around packet center (default 2.0)")
    args = parser.parse_args()

    basename = args.basename
    cachepath = args.cachepath
    channel = args.channel
    packet_idx = args.packet
    run = args.run
    date = args.date
    signal_kind = args.signal_kind
    window_s = args.window

    # --- Load data and plot ---
    t_rel, y_raw_win, y_filt_win, y_filt_full, fs = load_raw_and_filtered_window(
        basename=basename,
        cachepath=cachepath,
        date=date,
        run=run,
        channel=channel,
        packet_idx=packet_idx,
        window_s=window_s,
        signal_kind=signal_kind,
        step_s=DEFAULT_STEP_S,
    )

    plot_all(
        t_rel,
        y_raw_win,
        y_filt_win,
        y_filt_full,
        fs,
        channel=channel,
        packet_idx=packet_idx,
        signal_kind=signal_kind,
    )


if __name__ == "__main__":
    main()
