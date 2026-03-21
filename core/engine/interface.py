# type: ignore  # taichi ndarray types are not resolvable by mypy

import warnings
from typing import Optional

import numpy as np
import osmnx as ox
import rasterio.features
import scipy.stats as ss
import taichi as ti
from rasterio.transform import from_bounds

from core.engine.pathfinding import weighted_astar
from core.engine.physics import compute_risk_kernel


def bounds_from_points(
    start: tuple[float, float],
    end: tuple[float, float],
    padding: float = 0.2,
) -> dict[str, float]:
    """Build a bounding box around two lat/lon points with proportional padding."""
    lat_min = min(start[0], end[0])
    lat_max = max(start[0], end[0])
    lon_min = min(start[1], end[1])
    lon_max = max(start[1], end[1])

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    # Ensure a minimum span so two close points still produce a usable grid
    lat_pad = max(lat_span * padding, 0.002)
    lon_pad = max(lon_span * padding, 0.002)

    return {
        "south": lat_min - lat_pad,
        "north": lat_max + lat_pad,
        "west": lon_min - lon_pad,
        "east": lon_max + lon_pad,
    }


def calculate_risk_area(
    bounds: dict[str, float],
    drone_params: dict[str, float],
) -> tuple[np.ndarray, list[list[float]]]:
    lat_diff = bounds["north"] - bounds["south"]
    lon_diff = bounds["east"] - bounds["west"]

    rows = max(int((lat_diff * 111_000) / 10), 20)
    cols = max(int((lon_diff * 111_000 * np.cos(np.radians(bounds["south"]))) / 10), 20)
    raster_shape = (rows, cols)

    lat_dist = lat_diff * 111.0
    lon_dist = lon_diff * 111.0 * np.cos(np.radians(bounds["south"]))
    area_sq_km = lat_dist * lon_dist

    population_grid: np.ndarray

    if area_sq_km > 25.0:
        warnings.warn(
            f"Area too large ({area_sq_km:.1f} sq km, limit 25). Using uniform fallback.",
            stacklevel=2,
        )
        population_grid = np.ones(raster_shape, dtype=np.float32)
    else:
        population_grid = _fetch_building_grid(bounds, raster_shape)

    wind_rad = np.radians(drone_params["wind_dir"])
    shift_y = int(drone_params["wind_speed"] * np.cos(wind_rad) * (drone_params["alt"] / 10))
    shift_x = int(drone_params["wind_speed"] * np.sin(wind_rad) * (drone_params["alt"] / 10))

    variance = max(drone_params["alt"] * 0.5, 5.0)
    cov_matrix = [[variance, 0], [0, variance]]

    offset_y, offset_x = rows // 2, cols // 2
    pdf_center_y = offset_y + shift_y
    pdf_center_x = offset_x + shift_x

    x, y = np.mgrid[0:rows, 0:cols]
    eval_grid = np.vstack((x.ravel(), y.ravel())).T
    pdf = ss.multivariate_normal([pdf_center_y, pdf_center_x], cov_matrix).pdf(eval_grid)
    pdf = pdf.reshape(raster_shape).astype(np.float32)

    padded_pdf = np.zeros(((rows * 3) + 1, (cols * 3) + 1), dtype=np.float32)
    padded_pdf[rows:rows * 2, cols:cols * 2] = pdf

    sm_premult = (population_grid * drone_params["width"]).astype(np.float32)

    padded_centre_y = rows + offset_y
    padded_centre_x = cols + offset_x

    risk_map = np.zeros(raster_shape, dtype=np.float32)
    compute_risk_kernel(padded_pdf, sm_premult, risk_map, padded_centre_y, padded_centre_x)
    ti.sync()

    bounds_list = [[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]]
    return risk_map, bounds_list


def find_optimal_path(
    risk_map: np.ndarray,
    bounds: dict[str, float],
    start_latlon: tuple[float, float],
    end_latlon: tuple[float, float],
    risk_weight: float,
) -> Optional[list[tuple[float, float]]]:
    """Translate lat/lon waypoints to grid indices, run A*, return lat/lon path."""
    rows, cols = risk_map.shape

    rm_min = float(risk_map.min())
    rm_max = float(risk_map.max())
    risk_norm = (risk_map - rm_min) / (rm_max - rm_min + 1e-9)

    def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
        row = int((bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * rows)
        col = int((lon - bounds["west"]) / (bounds["east"] - bounds["west"]) * cols)
        return (max(0, min(rows - 1, row)), max(0, min(cols - 1, col)))

    def grid_to_latlon(row: int, col: int) -> tuple[float, float]:
        lat = bounds["north"] - (row / rows) * (bounds["north"] - bounds["south"])
        lon = bounds["west"] + (col / cols) * (bounds["east"] - bounds["west"])
        return (lat, lon)

    start_grid = latlon_to_grid(start_latlon[0], start_latlon[1])
    end_grid = latlon_to_grid(end_latlon[0], end_latlon[1])

    pixel_path = weighted_astar(risk_norm, start_grid, end_grid, risk_weight)
    if pixel_path is None:
        return None

    return [grid_to_latlon(r, c) for r, c in pixel_path]


def _fetch_building_grid(
    bounds: dict[str, float],
    raster_shape: tuple[int, int],
) -> np.ndarray:
    rows, cols = raster_shape
    bbox = (bounds["west"], bounds["south"], bounds["east"], bounds["north"])
    try:
        try:
            buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
        except TypeError:
            buildings = ox.features_from_bbox(
                bounds["north"], bounds["south"], bounds["east"], bounds["west"],
                tags={"building": True},
            )

        if buildings.empty:
            print("No buildings found in area; using uniform fallback.")
            return np.ones(raster_shape, dtype=np.float32)

        transform = from_bounds(
            bounds["west"], bounds["south"], bounds["east"], bounds["north"],
            cols, rows,
        )
        geoms = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].geometry
        if geoms.empty:
            return np.ones(raster_shape, dtype=np.float32)

        shapes = ((geom, 50.0) for geom in geoms)
        grid = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=raster_shape,
            transform=transform,
            fill=1.0,
            dtype=np.float32,
        )
        print(f"Loaded {len(geoms)} building geometries from OSM.")
        return grid

    except Exception as exc:
        print(f"OSM fetch failed ({exc}); using uniform fallback.")
        return np.ones(raster_shape, dtype=np.float32)
