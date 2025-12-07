import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import argparse
from pymap3d import enu2geodetic, geodetic2enu


from dasprocessor.debugging import  visualize_doa_fit, depth_correct_timestamp
from dasprocessor.channel_gps import compute_channel_positions
import json
from dasprocessor.delete_check.doa_results_io import DoaResult, append_doa_result
import os



def fit_doa(times, channel_positions_enu):
    """
    Fit DOA from arrival times and channel positions (ENU).

    """

    times = np.asarray(times, dtype=float)
    channel_positions_enu = np.asarray(channel_positions_enu, dtype=float).copy()


    # # Array axis (horizontal) from endpoints
    # axis = channel_positions_enu[-1] - channel_positions_enu[0]
    # axis /= np.linalg.norm(axis)

    # Center positions
    center = channel_positions_enu.mean(axis=0)           
    P = channel_positions_enu - center                     

    # SVD to find principal direction
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    axis = Vt[0]                                   

    # Normalize axis, should be unit vector already from SVD but just in case
    axis = axis / np.linalg.norm(axis)

    # enforce a consistent direction (e.g. roughly from first to last)
    if np.dot(axis, channel_positions_enu[-1] - channel_positions_enu[0]) < 0:
        axis = -axis
   

    s = channel_positions_enu @ axis
    s = s - s[0]        # make first sensor s=0
    X = s.reshape(-1, 1)
    y = times

    # 3) Robust or plain 1D regression: t = a * s + b
   
    base = LinearRegression()
    model = RANSACRegressor(
        base,
        min_samples=3,
        max_trials=200,
    )
    model.fit(X, y)
    slope = float(model.estimator_.coef_[0])   # dt/ds

    # Time residuals from this 1D model
    residuals = y - model.predict(X)


    return slope, residuals, model, axis


def cone_radian_from_slope(slope, speed=1475.0):
    axis_dot_n = speed * slope
    axis_dot_n = float(np.clip(axis_dot_n, -1.0, 1.0))
    angle_rad = np.arccos(axis_dot_n)

    return angle_rad



def orthonormal_basis_from_axis(axis):
    """
    Given a unit vector 'axis', build an orthonormal basis (e1, e2, e3),
    where e1 = axis, and e2, e3 span the perpendicular plane.
    """
    axis = np.asarray(axis, dtype=float)

    # Pick a vector not parallel to axis
    if abs(axis[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])

    e2 = np.cross(axis, tmp)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(axis, e2)

    return axis, e2, e3
'''

def cone_plane_intersection(
    channel_positions_enu,
    slope,
    source_depth,
    array_axis,
    angle_uncertainty_deg,
    speed=1475.0,
    n_samples=360,
):
    """
    Compute intersection curves between the DOA cone (from a 1D array)
    and the horizontal plane z = source_depth, assuming a plane wave.

    Returns THREE sets of ENU points:
    - inner_points:  cone with angle (theta - angle_uncertainty_deg)
    - nominal_points: cone with angle theta
    - outer_points:  cone with angle (theta + angle_uncertainty_deg)

    All returned arrays have shape (Mi, 3), where Mi can be 0 if no
    valid intersection for that cone.
    """

    channel_positions_enu = np.asarray(channel_positions_enu, dtype=float)
    array_ref = channel_positions_enu.mean(axis=0)

    # --- Nominal angle between array axis and propagation direction ---
    theta_nominal = cone_radian_from_slope(slope, speed=speed)  # radians
    theta_nominal_deg = np.degrees(theta_nominal)

    # --- Inner/outer cone angles in degrees, clipped to [0, 180] ---
    inner_deg = max(0.0, theta_nominal_deg - angle_uncertainty_deg)
    outer_deg = min(180.0, theta_nominal_deg + angle_uncertainty_deg)

    theta_inner = np.radians(inner_deg)
    theta_outer = np.radians(outer_deg)

    # --- Local orthonormal basis: e1 = array_axis ---
    e1, e2, e3 = orthonormal_basis_from_axis(array_axis)

    # --- Helper: intersect cone with given theta with the depth plane ---
    def intersect_for_theta(theta_val: float) -> np.ndarray:
        """
        Given a cone half-angle theta_val (radians) around e1,
        intersect with z = source_depth plane and return ENU points.
        """

        # Sample φ around the cone circle
        phi = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        cos_theta = np.cos(theta_val)
        sin_theta = np.sin(theta_val)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # PROPAGATION directions on cone:
        # n_prop(φ) = cosθ e1 + sinθ (cosφ e2 + sinφ e3)
        n_prop = (
            cos_theta * e1[None, :]
            + sin_theta * (
                cos_phi[:, None] * e2[None, :] + sin_phi[:, None] * e3[None, :]
            )
        )

        # We want directions FROM array TO source → flip sign
        n_vecs = -n_prop

        dz = source_depth - array_ref[2]
        n_z = n_vecs[:, 2]

        eps = 1e-8
        valid = np.abs(n_z) > eps  # avoid directions parallel to plane

        if not np.any(valid):
            return np.empty((0, 3))

        R = np.empty_like(n_z)
        R[valid] = dz / n_z[valid]

        # Only keep source positions in front of the array along n_vecs
        valid = valid & (R > 0)

        if not np.any(valid):
            return np.empty((0, 3))

        points = array_ref[None, :] + R[valid, None] * n_vecs[valid, :]
        return points

    # --- Compute intersections for inner, nominal, outer cones ---
    inner_points   = intersect_for_theta(theta_inner)
    nominal_points = intersect_for_theta(theta_nominal)
    outer_points   = intersect_for_theta(theta_outer)

    return inner_points, nominal_points, outer_points
'''
def cone_plane_intersection(
    channel_positions_enu,
    slope,
    source_depth,
    array_axis,
    angle_uncertainty_deg,
    speed=1475.0,
    n_samples=360,
    theta_eps_deg=0.5,   # avoid exactly 0 or 180
    max_range_m=None,    # NEW: max distance from array_ref in meters
    xy_window=None,      # NEW: (E_min, E_max, N_min, N_max) in ENU
):
    """
    Same as before, but now you can limit the returned points by:
      - max_range_m: distance from array_ref (ENU)
      - xy_window: (E_min, E_max, N_min, N_max) in ENU
    """

    channel_positions_enu = np.asarray(channel_positions_enu, dtype=float)
    array_ref = channel_positions_enu.mean(axis=0)

    # --- Nominal angle between array axis and propagation direction ---
    theta_nominal = cone_radian_from_slope(slope, speed=speed)  # radians
    theta_nominal_deg = np.degrees(theta_nominal)

    # Clamp nominal angle away from 0 and 180 to avoid exact degeneracy
    theta_nominal_deg = np.clip(
        theta_nominal_deg, theta_eps_deg, 180.0 - theta_eps_deg
    )

    # Inner/outer cone angles
    inner_deg = np.clip(
        theta_nominal_deg - angle_uncertainty_deg,
        theta_eps_deg,
        180.0 - theta_eps_deg,
    )
    outer_deg = np.clip(
        theta_nominal_deg + angle_uncertainty_deg,
        theta_eps_deg,
        180.0 - theta_eps_deg,
    )

    theta_inner   = np.radians(inner_deg)
    theta_outer   = np.radians(outer_deg)
    theta_nominal = np.radians(theta_nominal_deg)

    # --- Local orthonormal basis: e1 = array_axis ---
    e1, e2, e3 = orthonormal_basis_from_axis(array_axis)

    def intersect_for_theta(theta_val: float) -> np.ndarray:
        """
        Given a cone half-angle theta_val (radians) around e1,
        intersect with z = source_depth plane and return ENU points.
        Applies max_range_m / xy_window filters if given.
        """

        phi = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        cos_theta = np.cos(theta_val)
        sin_theta = np.sin(theta_val)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # PROPAGATION directions on cone:
        n_prop = (
            cos_theta * e1[None, :]
            + sin_theta * (
                cos_phi[:, None] * e2[None, :]
                + sin_phi[:, None] * e3[None, :]
            )
        )

        # Start with directions FROM array TO source
        n_vecs = -n_prop

        dz = source_depth - array_ref[2]
        eps = 1e-8

        def try_with(n_vecs_local):
            n_z = n_vecs_local[:, 2]
            valid = np.abs(n_z) > eps
            if not np.any(valid):
                return None

            R = np.empty_like(n_z)
            R[valid] = dz / n_z[valid]
            valid = valid & (R > 0)
            if not np.any(valid):
                return None

            pts = array_ref[None, :] + R[valid, None] * n_vecs_local[valid, :]

            # --- NEW: apply range filter (max distance from array_ref) ---
            if max_range_m is not None:
                d = np.linalg.norm(pts - array_ref[None, :], axis=1)
                mask = d <= max_range_m
                pts = pts[mask]
                if pts.shape[0] == 0:
                    return None

            # --- NEW: apply ENU window filter ---
            if xy_window is not None:
                E_min, E_max, N_min, N_max = xy_window
                E = pts[:, 0]
                N = pts[:, 1]
                mask = (
                    (E >= E_min) & (E <= E_max) &
                    (N >= N_min) & (N <= N_max)
                )
                pts = pts[mask]
                if pts.shape[0] == 0:
                    return None

            return pts

        # Try current hemisphere
        pts = try_with(n_vecs)
        if pts is not None:
            return pts

        # Try flipped hemisphere
        pts_flipped = try_with(-n_vecs)
        if pts_flipped is not None:
            return pts_flipped

        return np.empty((0, 3))

    # --- Compute intersections for inner, nominal, outer cones ---
    inner_points   = intersect_for_theta(theta_inner)
    nominal_points = intersect_for_theta(theta_nominal)
    outer_points   = intersect_for_theta(theta_outer)

    return inner_points, nominal_points, outer_points



def main() -> None:
    print("starting DOA computation...")

    parser = argparse.ArgumentParser(description="compute and visualize DOA from packet arrivals")
    parser.add_argument("--packet", type=int, default=0, help="Packet index to process")
    parser.add_argument("--start-channel", type=int, default=100, help="Start channel  of subarray")
    parser.add_argument("--array-length", type=int, default=25, help="Length of the subarray")
    args = parser.parse_args()

    from dasprocessor.plot.source_track import load_source_points_for_run, _tx_datetimes_for_run, _interp_positions_at_times
    from pathlib import Path
    import matplotlib.pyplot as plt
    


    fs = 25000  # Sampling frequency
    c = 1475.0  # m/s in water


    # 1. Load geodetic positions
    geodetic_channel_positions = compute_channel_positions(
        r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json",
        channel_count=1200,
        channel_distance=1.02
    )

    # 2. Load arrivals for all packets and all channels
    peaks_file = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-reordered_all_hilbert_channels.json"

    with open(
        peaks_file,
        'r'
    ) as file:
        packet_arrivals = json.load(file)



    packet_idx = args.packet
    all_channel_arrivals = packet_arrivals[str(args.packet)]

    # Build dict with int keys
    arrivals_dict = {
        ch: all_channel_arrivals[str(ch)]
        for ch in range(args.start_channel, args.start_channel + args.array_length)
        if str(ch) in all_channel_arrivals
    }

    # Channels present in THIS packet (already ints)
    channels = sorted(arrivals_dict.keys())

    # Use int keys here
    times_packet = np.array([arrivals_dict[ch] for ch in channels])

    
    # 2. Convert geodetics → ENU
    ref = geodetic_channel_positions.get(0)
    ref[2] = 0.0  # set ref altitude to zero, ocean level, the 0 channel is at 10 meters depth, mostly for plotting

    print(f"Reference position for ENU conversion: lat={ref[0]}, lon={ref[1]}, alt={ref[2]}")


    channel_positions_enu = []
    for ch in channels:
        channel_positions_enu.append(
            geodetic2enu(
                geodetic_channel_positions[ch][0],
                geodetic_channel_positions[ch][1],
                geodetic_channel_positions[ch][2],
                ref[0],
                ref[1],
                ref[2]
            )
        )
    channel_positions_enu = np.array(channel_positions_enu, dtype=float)

    # 3. Depth-correct arrival times, only for validation purposes
    # times_corr_sec = []
    # for ch, t in zip(channels, times_packet):
    #     t_corr = depth_correct_timestamp(packet_idx, ch, t, fs)
    #     times_corr_sec.append(t_corr)

    # times_corr_sec = np.array(times_corr_sec)


    
    times_sec = times_packet / fs
    

    # 4. Fit DOA
    slope, residuals, model, axis = fit_doa(times_sec, channel_positions_enu)

    max_extra_dist_1 = 10.0  # m
    tau_1 = max_extra_dist_1 / c  # ~6.8e-3 s
    mask_1 = np.abs(residuals) < tau_1

    channels_inlier = [ch for ch, keep in zip(channels, mask_1) if keep]
    positions_inlier = channel_positions_enu[mask_1]
    times_inlier = times_sec[mask_1]

    # Re-fit DOA using only inliers
    slope_refit, residuals_refit, model_refit, axis_refit = fit_doa(times_inlier, positions_inlier)

    

    # --- second-pass masking based on refit residuals ---

    max_extra_dist_2 = 5.0  # m
    tau_2 = max_extra_dist_2 / c  # ~3.4e-3 s
    mask_refit = np.abs(residuals_refit) < tau_2

    channels_inlier2   = [ch for ch, keep in zip(channels_inlier, mask_refit) if keep]
    positions_inlier2  = positions_inlier[mask_refit]
    times_inlier2      = times_inlier[mask_refit]
    residuals_refit2   = residuals_refit[mask_refit]


    # Optionally refit a third time on the doubly-cleaned data
    slope_final, residuals_final, model_final, axis_final = fit_doa(times_inlier2, positions_inlier2)

    print(f"Final slope after cleaning: {slope_final:.3e} s/m")


    cone_angle= cone_radian_from_slope(slope_final, speed=c)

    if np.degrees(cone_angle) <= 90:
        angle_off_axis = np.degrees(cone_angle)
        side = "+axis (first to last)"
    else:
        angle_off_axis = 180.0 - np.degrees(cone_angle)
        side = "-axis (last to first)"

    print(f"Angle between propagation and array axis: {np.degrees(cone_angle):.1f}°")
    print(f"Wave is {angle_off_axis:.1f}° off the axis, arriving from the {side} end.")


    # --- Compute ellipse at source depth ---
    source_depth = -30.0  # for example, ENU z = -50 m

    uncertainty_deg = 5.0  # e.g., 5 degrees uncertainty

    inner_points, nominal_points, outer_points  = cone_plane_intersection(
        positions_inlier2,
        slope=slope_final,
        source_depth=source_depth,
        speed=1475.0,
        n_samples=300,   # finer sampling if you want
        array_axis=axis_final,
        angle_uncertainty_deg=uncertainty_deg,  # e.g., 5 degrees uncertainty
        xy_window=(0, 700, -500, 500),
    )


    #dump the points into a json file

    savepath = Path(__file__).resolve().parent / f"../resources/subarray_ellipses/ellipse_points_start_ch_{args.start_channel}_pkt_{args.packet}_arrlength_{args.array_length}_unc_{uncertainty_deg}.json"
    savepath = savepath.resolve()
    os.makedirs(savepath.parent, exist_ok=True)

    ellipse_dict = {
        "inner_points": inner_points.tolist(),
        "nominal_points": nominal_points.tolist(),
        "outer_points": outer_points.tolist(),
    }
    
    with open(savepath, "w") as f:
        json.dump(ellipse_dict, f)

    print(f"Saved ellipse points to {savepath}")

    

    #center_enu = positions_inlier2[positions_inlier2.shape[0]//2]

    #ellipse_enu = cone_plane_intersection_enu(center_enu, axis_final, bearing_angle, z_source=-30.0)

    # latlon = []
    # for e, n, u in ellipse_enu:
    #     lat, lon, _ = enu2geodetic(e, n, u, ref[0], ref[1], ref[2])
    #     latlon.append([float(lat), float(lon)])






    # flat_pos = positions_inlier2.copy()
    # flat_pos[:, 2] = 0.0

    # mid_array_axis = flat_pos[len(flat_pos)//2] - flat_pos[len(flat_pos)//2+1]
    # mid_array_axis /= np.linalg.norm(mid_array_axis)

    # print(f"Mid array axis: {mid_array_axis}")
    # direction_1 = np.cos(bearing_angle)*mid_array_axis + np.sin(bearing_angle)*np.cross(np.array([0,0,1]), mid_array_axis)
    # direction_2 = np.cos(bearing_angle)*mid_array_axis - np.sin(bearing_angle)*np.cross(np.array([0,0,1]), mid_array_axis)
    # direction_source_A = direction_1/np.linalg.norm(direction_1)
    # direction_source_B = direction_2/np.linalg.norm(direction_2)
 

    # center_lat, center_lon, _ = enu2geodetic(flat_pos[len(flat_pos)//2][0], flat_pos[len(flat_pos)//2][1],0, ref[0], ref[1], ref[2])
 


    # doa_result = DoaResult(
    #     packet=packet_idx,
    #     center_lat=center_lat,
    #     center_lon=center_lon,
    #     dir_A_enu=direction_source_A.tolist(),
    #     dir_B_enu=direction_source_B.tolist(),
    #     channels_min=int(min(channels_inlier)),
    #     channels_max=int(max(channels_inlier)),
    #     n_channels=len(channels_inlier),
    #     ellipse_latlon=latlon,
    # )

    # savepath = Path(__file__).resolve().parent / f"../resources/B_4/DOA_results-{args.start_channel}-{args.start_channel + args.array_length}.json"
    # savepath = savepath.resolve()
    # os.makedirs(savepath.parent, exist_ok=True)


    # append_doa_result(savepath, doa_result)
    # print(f"Appended DOA result for packet {packet_idx} to {savepath}")



  
    #5. Visualize

    # axis = channel_positions_enu[-1] - channel_positions_enu[0]
    # axis /= np.linalg.norm(axis)

    # # project each channel position onto the axis
    # proj = channel_positions_enu @ axis  # 1D coordinate along-array

    # diffs = np.diff(proj)
    # print("mean spacing along local axis:", np.mean(diffs))

    # plt.figure()
    # plt.plot(proj, times_sec, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position before cleaning")
    # plt.show()


    # axis_1 = positions_inlier2[-1]-positions_inlier2[0]
    # axis_1 /= np.linalg.norm(axis_1)

    # proj_inlier = positions_inlier @ axis_1
    # plt.figure()
    # plt.plot(proj_inlier, times_inlier, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position after first cleaning")
    # plt.show()

    # axis_2 = positions_inlier2[-1]-positions_inlier2[0]
    # axis_2 /= np.linalg.norm(axis_2)

    # proj_inlier2 = positions_inlier2 @ axis_final
    # plt.figure()
    # plt.plot(proj_inlier2, times_inlier2, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position after second cleaning")
    # plt.show()


    
    # visualize_doa_fit(
    #     channel_positions_enu,
    #     times_corr_sec,
    #     residuals,
    #     direction_,
    #     left_right_ambiguity(direction, channel_positions_enu),
    #     channels,
    #     model,
    #     title=f"Packet {packet_idx} first pass"
    # )

    # visualize_doa_fit(
    #     positions_inlier,
    #     times_inlier,
    #     residuals_refit,
    #     direction_refit,
    #     left_right_ambiguity(direction_refit, positions_inlier),
    #     channels_inlier,
    #     model_refit,
    #     title=f"Packet {packet_idx} second pass"
    # )

#     direction_source_A = [0,0,1]
#     direction_source_B = [0,0,-1]

#     visualize_doa_fit(
#         positions_inlier2,
#         times_inlier2,
#         residuals_refit2,
#         direction_source_A,
#         direction_source_B,
#         channels_inlier2,
#         model_final,
#         title=f"Packet {packet_idx} final fit"
# )



if __name__ == "__main__":
    main()  
