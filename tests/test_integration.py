import os
import requests
# pyrefly: ignore [missing-import]
import pytest
import numpy as np

from core.engine.interface import calculate_risk_area, find_optimal_path

def test_end_to_end_calculation_and_simulation():
    """
    Programmatic integration test:
    1. Runs the core ground risk calculation.
    2. Computes the optimal path.
    3. Posts the path waypoints to the running FastAPI simulation server.
    4. Verifies the telemetry output format.
    """
    # 1. Setup coordinates & drone parameters
    start_pt = (51.3360, -0.2670)
    end_pt = (51.3380, -0.2680)
    
    # Calculate bounding box
    center_lat = (start_pt[0] + end_pt[0]) / 2.0
    center_lon = (start_pt[1] + end_pt[1]) / 2.0
    lat_diff = abs(start_pt[0] - end_pt[0])
    lon_diff = abs(start_pt[1] - end_pt[1])
    radius_deg = max(lat_diff / 2.0, lon_diff / 2.0) + 0.008
    bounds = {
        "north": center_lat + radius_deg,
        "south": center_lat - radius_deg,
        "east":  center_lon + radius_deg,
        "west":  center_lon - radius_deg,
    }
    
    drone_params = {
        "mass": 0.9,
        "width": 0.35,
        "speed": 15.0,
        "alt": 50.0,
        "wind_speed": 5.0,
        "wind_dir": 90,
    }
    
    # 2. Run risk calculation (using uniform mock for testing if offline, or real OSM/Taichi if available)
    # To keep integration test robust, we fetch or mock similarly
    try:
        risk_map, bounds_list = calculate_risk_area(bounds, drone_params)
    except Exception as exc:
        pytest.fail(f"Risk calculation failed: {exc}")
        
    # 3. Compute optimal path
    path = find_optimal_path(risk_map, bounds, start_pt, end_pt, risk_weight=10.0)
    assert path is not None, "Optimal path calculation failed"
    assert len(path) > 0
    
    # 4. Prepare payload for the simulation API
    waypoints = [{"lat": lat, "lon": lon, "alt": drone_params["alt"]} for lat, lon in path]
    payload = {
        "waypoints": waypoints,
        "speed": drone_params["speed"],
        "drone_params": drone_params
    }
    
    # 5. Send POST request to the local simulation API (running on port 8001 in our background task)
    sim_api_url = os.getenv("SIM_API_URL", "http://localhost:8001")
    
    try:
        response = requests.post(f"{sim_api_url}/simulate_path", json=payload, timeout=10)
        assert response.status_code == 200, f"Simulation API error: {response.text}"
        
        result = response.json()
        assert result["status"] == "completed"
        assert "telemetry" in result
        assert "summary" in result
        
        telemetry = result["telemetry"]
        assert len(telemetry) > 0
        assert "lat" in telemetry[0]
        assert "lon" in telemetry[0]
        assert "battery" in telemetry[0]
        
        summary = result["summary"]
        assert summary["total_flight_time_sec"] > 0
        assert summary["total_distance_flown_m"] > 0
        assert not summary["collision_detected"]
        
        print(f"\nIntegration test succeeded!")
        print(f"Path Waypoints: {len(path)}")
        print(f"Simulated Steps: {len(telemetry)}")
        print(f"Flight Duration: {summary['total_flight_time_sec']}s")
        print(f"Distance Flown: {summary['total_distance_flown_m']}m")
        print(f"Battery Used: {summary['battery_used_pct']}%")
        
    except requests.exceptions.ConnectionError:
        pytest.fail(f"Could not connect to simulation API at {sim_api_url}. Is it running?")
