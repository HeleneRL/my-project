from __future__ import annotations
from typing import Dict, Any, List, Tuple
import folium
import numpy as np
import math

# WGS-84 ellipsoid
_WGS84_A = 6378137.0              # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563    # flattening
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_WGS84_E2 = 1 - (_WGS84_B**2) / (_WGS84_A**2)





def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    sλ, cλ = math.sin(lon), math.cos(lon)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sL*sL)
    x = (N + h) * cL * cλ
    y = (N + h) * cL * sλ
    z = (N * (1 - _WGS84_E2) + h) * sL
    return np.array([x, y, z], float)

def ecef_to_enu_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    sL, cL = math.sin(lat0), math.cos(lat0)
    sλ, cλ = math.sin(lon0), math.cos(lon0)
    return np.array([
        [-sλ,        cλ,        0.0],
        [-sL*cλ,    -sL*sλ,     cL ],
        [ cL*cλ,     cL*sλ,     sL ]
    ], float)

def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = xyz.astype(float)
    a = _WGS84_A; b = _WGS84_B; e2 = _WGS84_E2
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    E2 = a*a - b*b
    F = 54 * b*b * z*z
    G = r*r + (1 - e2)*z*z - e2*E2
    c = (e2*e2 * F * r*r) / (G*G*G)
    s = (1 + c + math.sqrt(c*c + 2*c))**(1/3)
    P = F / (3 * (s + 1/s + 1)**2 * G*G)
    Q = math.sqrt(1 + 2*e2*e2*P)
    r0 = -(P*e2*r)/(1+Q) + math.sqrt(0.5*a*a*(1+1/Q) - (P*(1-e2)*z*z)/(Q*(1+Q)) - 0.5*P*r*r)
    U = math.sqrt((r - e2*r0)**2 + z*z)
    V = math.sqrt((r - e2*r0)**2 + (1 - e2)*z*z)
    z0 = (b*b * z) / (a * V)
    lat = math.atan2(z + (e2 * z0), r)
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    h = r / math.cos(lat) - N
    return (math.degrees(lat), math.degrees(lon), h)

def enu_offset_to_geodetic(offset_enu: np.ndarray,
                           origin_geo: Tuple[float, float, float]) -> Tuple[float, float, float]:
    lat0, lon0, h0 = origin_geo
    ecef_ref = geodetic_to_ecef(lat0, lon0, h0)
    R = ecef_to_enu_matrix(lat0, lon0)          # ENU = R @ (ECEF - ref)
    ecef_target = ecef_ref + R.T @ offset_enu   # invert
    return ecef_to_geodetic(ecef_target)



def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    """
    Convert geodetic (lat, lon in degrees; h in meters) to ECEF (m).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat**2)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - _WGS84_E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_enu_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """
    Rotation matrix R such that: ENU = R @ (ECEF - ECEF_ref)
    """
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sL, cL = math.sin(lat0), math.cos(lat0)
    sλ, cλ = math.sin(lon0), math.cos(lon0)

    # Rows are unit vectors of E, N, U expressed in ECEF
    R = np.array([
        [-sλ,            cλ,           0.0],
        [-sL*cλ,        -sL*sλ,        cL ],
        [ cL*cλ,         cL*sλ,        sL ]
    ], dtype=float)
    return R

def geodetic_to_enu(points: List[Tuple[float, float, float]],
                    ref: Tuple[float, float, float]) -> np.ndarray:
    """
    Convert a list of geodetic points to ENU (meters) using 'ref' (lat, lon, alt) as origin.
    Returns array shape (N, 3).
    """
    lat0, lon0, h0 = ref
    ecef_ref = geodetic_to_ecef(lat0, lon0, h0)
    R = ecef_to_enu_matrix(lat0, lon0)
    enu_list = []
    for (lat, lon, h) in points:
        ecef = geodetic_to_ecef(lat, lon, h)
        enu = R @ (ecef - ecef_ref)
        enu_list.append(enu)
    return np.vstack(enu_list)