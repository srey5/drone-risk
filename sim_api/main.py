import logging
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

try:
    from sim_api.omni_runner import simulate_flight  # run from project root: uvicorn sim_api.main:app
except ModuleNotFoundError:
    from omni_runner import simulate_flight  # run from inside sim_api/: uvicorn main:app


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sim_api")

app = FastAPI(
    title="OmniDrones Simulation API Server",
    description="A service to simulate UAV flight trajectories under environmental constraints.",
    version="1.0.0"
)

# Enable CORS for frontend/Streamlit interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Input Validation ---
class Waypoint(BaseModel):
    lat: float = Field(..., description="Latitude of the waypoint")
    lon: float = Field(..., description="Longitude of the waypoint")
    alt: Optional[float] = Field(None, description="Altitude of the waypoint in meters")

class SimulationRequest(BaseModel):
    waypoints: List[Waypoint] = Field(..., min_length=1, description="Sequential list of flight waypoints")
    speed: float = Field(15.0, gt=0.0, description="Cruise speed of the UAV in m/s")
    drone_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical drone specs (mass, width) and env settings (wind_speed, wind_dir, alt)"
    )

# --- Endpoints ---
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "sim_api"}

@app.post("/simulate_path")
def simulate_path(request: SimulationRequest):
    """
    Simulates drone flight path using Isaac Sim / OmniDrones physics.
    Accepts GPS waypoints, cruise speed, and drone parameters.
    Returns status, detailed time-series telemetry, and a summary.
    """
    logger.info(f"Received simulation request: {len(request.waypoints)} waypoints at {request.speed} m/s")
    
    # Convert Pydantic model to list of dictionaries
    wps_dict = []
    for wp in request.waypoints:
        wp_data = {"lat": wp.lat, "lon": wp.lon}
        if wp.alt is not None:
            wp_data["alt"] = wp.alt
        wps_dict.append(wp_data)
        
    try:
        # Run simulation
        result = simulate_flight(
            waypoints=wps_dict,
            speed=request.speed,
            drone_params=request.drone_params
        )
        return result
    except Exception as e:
        logger.error(f"Simulation failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")
