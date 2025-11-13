import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, peak_prominences

def _xcorr_normalized(x, h):
    """valid-mode, normalized cross-correlation."""
    xc = correlate(x, h, mode='valid')
    m = np.max(np.abs(xc))
    return xc / m if m != 0 else xc




def plot_channel_corr_and_peaks(rx_col, preamble, peak_properties,
                                targets=None, tol=None, fs=25_000,
                                zoom_center=None, zoom_halfwidth=500,
                                title_prefix=""):
    """
    Draw normalized correlation, overlay detected peaks (height & prominence),
    and optional target windows (target ± tol).

    rx_col          : 1-D array with one channel of DAS samples (already filtered)
    preamble        : 1-D array (same one you pass to detector)
    peak_properties : dict you pass to find_peaks (prominence, height, distance)
    targets         : 1-D array of expected packet sample indices (SAME timebase as peaks),
                      e.g. target = med_first + np.arange(N)*sequence_period
                      (but shifted to your slice’s origin if needed)
    tol             : integer (samples). If not None, draw target+tol bands.
    zoom_center     : center sample of a zoom window (in correlation index coords)
    zoom_halfwidth  : half-width of the zoom window (samples)
    """
    # 1) correlation (normalized)
    xc = _xcorr_normalized(rx_col, preamble)

    # 2) run the exact same peak finder you use in production
    from scipy.signal import find_peaks
    pk_idx, pk_props = find_peaks(xc, **peak_properties)

    # compute prominences explicitly (useful to see real values)
    prom, left_bases, right_bases = peak_prominences(xc, pk_idx)

    # 3) choose plotting range
    n = len(xc)
    if zoom_center is None:
        lo, hi = 0, n
    else:
        lo = max(0, int(zoom_center - zoom_halfwidth))
        hi = min(n, int(zoom_center + zoom_halfwidth))

    xs = np.arange(lo, hi)
    xcv = xc[lo:hi]

    # make the plot
    plt.figure(figsize=(11, 4))
    plt.plot(xs, xcv, label='normalized corr')
    # overlay peaks in range
    in_rng = (pk_idx >= lo) & (pk_idx < hi)
    pk = pk_idx[in_rng]
    if pk.size:
        plt.plot(pk, xc[pk], "o", label="peaks")
        # annotate height and prominence for a few nearest peaks (avoid clutter)
        for i, k in enumerate(pk[:20]):  # cap annotations
            txt = f"h={xc[k]:.2f}, p={prom[np.where(pk_idx==k)[0][0]]:.2f}"
            plt.annotate(txt, (k, xc[k]), xytext=(5, 8),
                         textcoords='offset points', fontsize=8)

    # 4) overlay target windows (target±tol) if provided
    if targets is not None and tol is not None:
        # IMPORTANT: targets must be in the SAME coordinate system as correlation indices
        # If your targets are in RAW sample indices, convert: target_corr = target_raw - PRE
        for t in targets:
            t_corr = t
            if lo <= t_corr < hi:
                plt.axvspan(t_corr-tol, t_corr + tol, alpha=0.15, color='gray')
                plt.axvline(t_corr, ls='--', alpha=0.4, label='target' if t == targets[0] else None)
                print(f"Target position (corr idx): {t_corr}")

    plt.xlabel("correlation index (samples)")
    plt.ylabel("normalized corr")
    plt.title(f"{title_prefix} corr & peaks")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_channel_corr_with_selected(
    rx_col,
    preamble,
    selected_peaks_map,
    *,
    targets=None,
    tol=None,
    fs=25_000,
    targets_are_raw=False,
    zoom_center=None,
    zoom_halfwidth=800,
    title_prefix="",
    annotate=True
):
    """
    Plot normalized matched-filter correlation for ONE channel and overlay:
      - targets (dashed line) + optional ±tol band, and
      - selected peaks (those that actually qualified and were saved).

    Parameters
    ----------
    rx_col : 1-D array
        One DAS channel (already filtered).
    preamble : 1-D array
        The reference preamble.
    selected_peaks_map : dict[int -> int]
        {packet_id : corr_index} for this channel (from mypeaks[ch]).
        NOTE: corr_index is in *correlation* coordinates.
    targets : 1-D array, optional
        Expected packet positions. If `targets_are_raw=False`, they are
        assumed to already be in correlation indices. If raw, set
        `targets_are_raw=True`.
    tol : int, optional
        Tolerance window (samples). If given, draw target ± tol band(s).
    fs : int
        Sample rate (for axis labels only).
    targets_are_raw : bool
        If True, convert targets to correlation coordinates by subtracting PRE.
    zoom_center : int, optional
        Center index (correlation coords) for zoom.
    zoom_halfwidth : int
        Half-width for zoom window (samples).
    title_prefix : str
        Title prefix (e.g., "ch 148").
    annotate : bool
        If True, annotate selected peaks with packet id and value.
    """
    PRE = len(preamble) - 1

    # 1) correlation (normalized)
    xc = _xcorr_normalized(rx_col, preamble)
    xc = np.sqrt(xc**2)  # use absolute value for plotting
    n = len(xc)

    # 2) convert targets if provided
    if targets is not None:
        targets = np.asarray(targets, dtype=int)
        if targets_are_raw:
            targets = targets - PRE  # convert to correlation indices

    # 3) pick plotting window
    if zoom_center is None:
        lo, hi = 0, n
    else:
        lo = max(0, int(zoom_center - zoom_halfwidth))
        hi = min(n, int(zoom_center + zoom_halfwidth))

    xs = np.arange(lo, hi)
    xcv = xc[lo:hi]

    # 4) plot
    plt.figure(figsize=(11, 4))
    plt.plot(xs, xcv, label="normalized corr")

    # 5) targets + tol bands
    if targets is not None:
        first_drawn = True
        for t in targets:
            if lo <= t < hi:
                if tol is not None:
                    plt.axvspan(t - tol, t + tol, alpha=0.12, label="target±tol" if first_drawn else None)
                plt.axvline(t, ls="--", alpha=0.5, label="target" if first_drawn else None)
                first_drawn = False

    # 6) selected peaks (packet_id -> corr_idx)
    if selected_peaks_map:
        pkt_ids = sorted(selected_peaks_map.keys())
        sel_idx = np.array([selected_peaks_map[p] for p in pkt_ids], dtype=int)
        in_rng = (sel_idx >= lo) & (sel_idx < hi)
        if np.any(in_rng):
            plt.plot(sel_idx[in_rng], xc[sel_idx[in_rng]], "o", label="selected peaks")
            if annotate:
                for pid, idx in zip(np.array(pkt_ids)[in_rng], sel_idx[in_rng]):
                    plt.annotate(f"pkt {pid}\n{idx}", (idx, xc[idx]),
                                 xytext=(6, 8), textcoords="offset points", fontsize=8)

    plt.xlabel("correlation index (samples)")
    plt.ylabel("normalized corr")
    plt.title(f"{title_prefix} corr + selected peaks (fs={fs} Hz)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()