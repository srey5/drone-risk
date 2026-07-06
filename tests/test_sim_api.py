# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from sim_api.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint returns 200 and correct body."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "sim_api"}

def test_simulate_path_success():
    """Test standard simulation call with valid parameters."""
    payload = {
        "waypoints": [
            {"lat": 51.3360, "lon": -0.2670, "alt": 50.0},
            {"lat": 51.3380, "lon": -0.2680, "alt": 50.0}
        ],
        "speed": 12.0,
        "drone_params": {
            "mass": 1.5,
            "width": 0.6,
            "wind_speed": 4.0,
            "wind_dir": 180,
            "alt": 50.0
        }
    }
    response = client.post("/simulate_path", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "completed"
    assert "telemetry" in data
    assert "summary" in data
    
    telemetry = data["telemetry"]
    assert len(telemetry) > 0
    # Verify starting coordinates match start point
    assert telemetry[0]["lat"] == 51.3360
    assert telemetry[0]["lon"] == -0.2670
    
    summary = data["summary"]
    assert summary["total_flight_time_sec"] > 0
    assert summary["total_distance_flown_m"] > 0
    assert summary["average_speed_m_s"] > 0
    assert summary["battery_used_pct"] > 0
    assert summary["max_deviation_m"] >= 0
    assert isinstance(summary["collision_detected"], bool)

def test_simulate_path_missing_waypoints():
    """Test that empty or missing waypoints causes validation failure."""
    # Empty waypoints list
    payload = {
        "waypoints": [],
        "speed": 15.0
    }
    response = client.post("/simulate_path", json=payload)
    assert response.status_code == 422
    
    # Missing waypoints entirely
    payload_missing = {
        "speed": 15.0
    }
    response_missing = client.post("/simulate_path", json=payload_missing)
    assert response_missing.status_code == 422

def test_simulate_path_invalid_speed():
    """Test that zero or negative speed causes validation failure."""
    payload = {
        "waypoints": [
            {"lat": 51.3360, "lon": -0.2670, "alt": 50.0}
        ],
        "speed": -5.0
    }
    response = client.post("/simulate_path", json=payload)
    assert response.status_code == 422
