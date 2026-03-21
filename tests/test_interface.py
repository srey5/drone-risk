"""Tests for core/engine/interface.py.

OSM fetching and the Taichi GPU kernel are patched out so tests stay fast
and offline. Only the pure-Python logic (bounds math, lat/lon ↔ grid
conversion, risk normalisation) is exercised directly.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.engine.interface import bounds_from_points, calculate_risk_area, find_optimal_path


# ---------------------------------------------------------------------------
# bounds_from_points
# ---------------------------------------------------------------------------

class TestBoundsFromPoints:
    def test_basic_ordering(self):
        bounds = bounds_from_points((51.0, -0.5), (51.5, 0.0))
        assert bounds["south"] < bounds["north"]
        assert bounds["west"] < bounds["east"]

    def test_swap_gives_same_result(self):
        a, b = (51.0, -0.5), (51.5, 0.0)
        b1 = bounds_from_points(a, b)
        b2 = bounds_from_points(b, a)
        for key in ("south", "north", "west", "east"):
            assert b1[key] == pytest.approx(b2[key])

    def test_padding_expands_bounds(self):
        """Padded bounds must strictly contain the raw extent."""
        start, end = (51.0, -0.5), (51.5, 0.0)
        bounds = bounds_from_points(start, end, padding=0.2)
        assert bounds["south"] < min(start[0], end[0])
        assert bounds["north"] > max(start[0], end[0])
        assert bounds["west"] < min(start[1], end[1])
        assert bounds["east"] > max(start[1], end[1])

    def test_minimum_span_for_coincident_points(self):
        """Two identical points should still produce a non-degenerate bbox."""
        bounds = bounds_from_points((51.5, -0.1), (51.5, -0.1))
        assert bounds["north"] - bounds["south"] > 0
        assert bounds["east"] - bounds["west"] > 0

    def test_minimum_span_value(self):
        """The minimum guaranteed span is 2 × 0.002 = 0.004 degrees."""
        bounds = bounds_from_points((51.5, -0.1), (51.5, -0.1))
        assert (bounds["north"] - bounds["south"]) >= 0.004
        assert (bounds["east"] - bounds["west"]) >= 0.004

    def test_larger_padding_gives_larger_bbox(self):
        a, b = (51.0, -1.0), (52.0, 0.0)
        small = bounds_from_points(a, b, padding=0.1)
        large = bounds_from_points(a, b, padding=0.5)
        assert large["north"] > small["north"]
        assert large["south"] < small["south"]
        assert large["east"] > small["east"]
        assert large["west"] < small["west"]

    def test_all_keys_present(self):
        bounds = bounds_from_points((51.0, -0.5), (51.5, 0.0))
        assert set(bounds.keys()) == {"south", "north", "west", "east"}


# ---------------------------------------------------------------------------
# calculate_risk_area  (OSM + Taichi mocked)
# ---------------------------------------------------------------------------

def _make_fake_kernel(padded_pdf, sm_premult, out_map, pcy, pcx):
    """Stand-in for compute_risk_kernel: fills out_map with a gradient."""
    rows, cols = out_map.shape
    for r in range(rows):
        for c in range(cols):
            out_map[r, c] = float(r + c) / (rows + cols)


SMALL_BOUNDS = {"south": 51.49, "north": 51.51, "west": -0.01, "east": 0.01}
DRONE_PARAMS = {"mass": 0.9, "width": 0.35, "speed": 15.0, "alt": 50.0,
                "wind_speed": 5.0, "wind_dir": 90}


@pytest.fixture
def mock_engine(monkeypatch):
    """Patch out OSM fetch and Taichi so tests run offline and fast."""
    monkeypatch.setattr(
        "core.engine.interface._fetch_building_grid",
        lambda bounds, shape: np.ones(shape, dtype=np.float32),
    )
    monkeypatch.setattr(
        "core.engine.interface.compute_risk_kernel",
        _make_fake_kernel,
    )
    monkeypatch.setattr("core.engine.interface.ti.sync", lambda: None)


class TestCalculateRiskArea:
    def test_returns_tuple_of_ndarray_and_list(self, mock_engine):
        result = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        risk_map, bounds_list = result
        assert isinstance(risk_map, np.ndarray)
        assert isinstance(bounds_list, list)

    def test_risk_map_is_2d(self, mock_engine):
        risk_map, _ = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        assert risk_map.ndim == 2

    def test_risk_map_minimum_size(self, mock_engine):
        risk_map, _ = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        assert risk_map.shape[0] >= 20
        assert risk_map.shape[1] >= 20

    def test_bounds_list_structure(self, mock_engine):
        _, bounds_list = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        # [[south, west], [north, east]]
        assert len(bounds_list) == 2
        assert len(bounds_list[0]) == 2
        assert len(bounds_list[1]) == 2

    def test_bounds_list_values_match_input(self, mock_engine):
        _, bounds_list = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        assert bounds_list[0][0] == pytest.approx(SMALL_BOUNDS["south"])
        assert bounds_list[0][1] == pytest.approx(SMALL_BOUNDS["west"])
        assert bounds_list[1][0] == pytest.approx(SMALL_BOUNDS["north"])
        assert bounds_list[1][1] == pytest.approx(SMALL_BOUNDS["east"])

    def test_large_area_uses_uniform_grid_and_warns(self, mock_engine, recwarn):
        large_bounds = {"south": 48.0, "north": 50.0, "west": -2.0, "east": 2.0}
        with pytest.warns(UserWarning, match="Area too large"):
            risk_map, _ = calculate_risk_area(large_bounds, DRONE_PARAMS)
        assert risk_map is not None

    def test_risk_map_dtype_is_float32(self, mock_engine):
        risk_map, _ = calculate_risk_area(SMALL_BOUNDS, DRONE_PARAMS)
        assert risk_map.dtype == np.float32

    def test_zero_wind_speed(self, mock_engine):
        params = {**DRONE_PARAMS, "wind_speed": 0.0}
        risk_map, _ = calculate_risk_area(SMALL_BOUNDS, params)
        assert risk_map is not None

    def test_high_altitude(self, mock_engine):
        params = {**DRONE_PARAMS, "alt": 400.0}
        risk_map, _ = calculate_risk_area(SMALL_BOUNDS, params)
        assert risk_map is not None


# ---------------------------------------------------------------------------
# find_optimal_path
# ---------------------------------------------------------------------------

class TestFindOptimalPath:
    """Tests for lat/lon ↔ grid conversion and path output format.

    Uses a small synthetic risk map so no OSM or Taichi calls are needed.
    """

    BOUNDS = {"south": 51.0, "north": 52.0, "west": -1.0, "east": 0.0}

    def _zero_risk_map(self, rows=50, cols=50):
        return np.zeros((rows, cols), dtype=np.float32)

    def test_returns_list_of_tuples(self):
        rm = self._zero_risk_map()
        start = (51.1, -0.9)
        end = (51.9, -0.1)
        path = find_optimal_path(rm, self.BOUNDS, start, end, risk_weight=0.0)
        assert path is not None
        assert isinstance(path, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in path)

    def test_path_endpoints_close_to_input(self):
        """First and last lat/lon should be near the requested start/end."""
        rm = self._zero_risk_map()
        start = (51.1, -0.9)
        end = (51.9, -0.1)
        path = find_optimal_path(rm, self.BOUNDS, start, end, risk_weight=0.0)
        # Grid resolution is 1/50 of the span — allow 1 cell of error
        lat_res = (self.BOUNDS["north"] - self.BOUNDS["south"]) / 50
        lon_res = (self.BOUNDS["east"] - self.BOUNDS["west"]) / 50
        assert abs(path[0][0] - start[0]) <= lat_res + 1e-6
        assert abs(path[0][1] - start[1]) <= lon_res + 1e-6
        assert abs(path[-1][0] - end[0]) <= lat_res + 1e-6
        assert abs(path[-1][1] - end[1]) <= lon_res + 1e-6

    def test_all_path_coords_within_bounds(self):
        rm = self._zero_risk_map()
        path = find_optimal_path(
            rm, self.BOUNDS, (51.1, -0.9), (51.9, -0.1), risk_weight=0.0
        )
        for lat, lon in path:
            assert self.BOUNDS["south"] <= lat <= self.BOUNDS["north"]
            assert self.BOUNDS["west"] <= lon <= self.BOUNDS["east"]

    def test_same_start_and_end(self):
        rm = self._zero_risk_map()
        point = (51.5, -0.5)
        path = find_optimal_path(rm, self.BOUNDS, point, point, risk_weight=0.0)
        assert path is not None
        assert len(path) == 1

    def test_out_of_bounds_points_clamped(self):
        """Points outside the bbox should be clamped and still return a path."""
        rm = self._zero_risk_map()
        path = find_optimal_path(
            rm, self.BOUNDS, (50.0, -2.0), (53.0, 1.0), risk_weight=0.0
        )
        assert path is not None

    def test_risk_weight_zero_vs_high_same_endpoints(self):
        rm = np.random.rand(30, 30).astype(np.float32)
        start, end = (51.1, -0.9), (51.9, -0.1)
        path_low = find_optimal_path(rm, self.BOUNDS, start, end, risk_weight=0.0)
        path_high = find_optimal_path(rm, self.BOUNDS, start, end, risk_weight=50.0)
        assert path_low is not None and path_high is not None
        # Both must reach the end regardless of weight
        lat_res = (self.BOUNDS["north"] - self.BOUNDS["south"]) / 30
        assert abs(path_low[-1][0] - path_high[-1][0]) <= lat_res + 1e-6
