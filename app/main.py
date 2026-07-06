import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import folium
# pyrefly: ignore [missing-import]
from folium.plugins import Draw
# pyrefly: ignore [missing-import]
from streamlit_folium import st_folium
# pyrefly: ignore [missing-import]
import matplotlib

from core.engine.interface import (
    calculate_risk_area,
    find_optimal_path,
)

# --- Page Configuration ---
st.set_page_config(page_title="UAS Pathfinding and Risk Analysis", layout="wide")

# --- CSS: remove default Streamlit chrome, tighten padding ---
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }

    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
for key in ("start_point", "end_point", "risk_map", "bounds_list", "bounds", "path_latlon", "sim_telemetry", "sim_summary"):
    if key not in st.session_state:
        st.session_state[key] = None

if "map_center" not in st.session_state:
    st.session_state["map_center"] = [51.3360, -0.2670]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 13
if "last_processed_drawing" not in st.session_state:
    st.session_state["last_processed_drawing"] = None

has_start = st.session_state["start_point"] is not None
has_end = st.session_state["end_point"] is not None

# --- Sidebar ---
st.sidebar.title("UAS Ground Risk Analyzer")

calculate_btn = st.sidebar.button(
    "Calculate Risk Map", type="primary", use_container_width=True
)

st.sidebar.markdown("---")

overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.6)
risk_weight = st.sidebar.slider("Risk Aversion", 0.0, 100.0, 10.0)
st.sidebar.caption("0 = shortest distance. 100 = strongly avoids high-risk areas.")

with st.sidebar.expander("Drone Specifications", expanded=False):
    drone_preset = st.selectbox(
        "Select Preset", ["Custom", "DJI Mavic 3", "DJI Matrice 300 RTK"]
    )
    mass, width, speed, alt = 1.0, 0.5, 15.0, 50.0
    if drone_preset == "DJI Mavic 3":
        mass, width, speed, alt = 0.9, 0.35, 15.0, 50.0
    elif drone_preset == "DJI Matrice 300 RTK":
        mass, width, speed, alt = 9.0, 0.81, 15.0, 100.0

    input_mass = st.number_input("Mass (kg)", min_value=0.1, max_value=50.0, value=mass)
    input_width = st.number_input("Width (m)", min_value=0.1, max_value=5.0, value=width)
    input_speed = st.number_input(
        "Cruise Speed (m/s)", min_value=1.0, max_value=50.0, value=speed
    )
    input_alt = st.number_input(
        "Cruise Altitude (m)", min_value=10.0, max_value=500.0, value=alt
    )

with st.sidebar.expander("Environmental Parameters", expanded=False):
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0)
    wind_dir = st.slider("Wind Direction (degrees)", 0, 360, 90)

st.sidebar.markdown("---")

simulate_btn = False
if st.session_state["path_latlon"] is not None:
    simulate_btn = st.sidebar.button(
        "Simulate in OmniDrones", type="primary", use_container_width=True
    )
    st.sidebar.markdown("---")

if st.sidebar.button("Reset Points"):
    for key in ("start_point", "end_point", "risk_map", "bounds_list", "bounds", "path_latlon", "sim_telemetry", "sim_summary"):
        st.session_state[key] = None
    st.session_state["last_processed_drawing"] = None
    st.rerun()

# --- Status caption ---
if has_start and has_end:
    st.caption("Start and End points set. Adjust parameters and click Calculate.")
elif has_start:
    st.caption("Start point set. Drop an End marker on the map.")
else:
    st.caption("Drop a Start marker and an End marker, then click Calculate.")

# --- Build Map ---
# Always restore the last known viewport so the map does not snap on rerun.
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=st.session_state["map_zoom"],
)

# Marker-only Draw plugin — all other tools disabled.
Draw(
    export=False,
    draw_options={
        "marker": True,
        "polygon": False,
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "circlemarker": False,
    },
).add_to(m)

# Re-inject persisted waypoints as Folium markers so they survive reruns.
if has_start:
    folium.Marker(
        location=st.session_state["start_point"],
        tooltip="A - Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
if has_end:
    folium.Marker(
        location=st.session_state["end_point"],
        tooltip="B - End",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

# Risk heatmap overlay
if st.session_state["risk_map"] is not None and st.session_state["bounds_list"] is not None:
    rm = st.session_state["risk_map"]
    rm_norm = (rm - rm.min()) / (rm.max() - rm.min() + 1e-9)

    colormap = matplotlib.colormaps["inferno"]
    color_image = colormap(rm_norm)
    color_image[:, :, 3] = rm_norm ** 0.5
    color_image[rm_norm < 0.05, 3] = 0.0

    folium.raster_layers.ImageOverlay(
        image=color_image,
        bounds=st.session_state["bounds_list"],
        name="Risk Heatmap",
        opacity=overlay_opacity,
    ).add_to(m)
    folium.LayerControl().add_to(m)

# Path overlay
if st.session_state["path_latlon"] is not None:
    folium.PolyLine(
        locations=st.session_state["path_latlon"],
        color="#00e5ff",
        weight=3,
        opacity=0.9,
        tooltip="Optimal Path",
    ).add_to(m)

# Simulated Path overlay
if st.session_state["sim_telemetry"] is not None:
    sim_coords = [(t["lat"], t["lon"]) for t in st.session_state["sim_telemetry"]]
    folium.PolyLine(
        locations=sim_coords,
        color="#ff9100",
        weight=3,
        opacity=0.85,
        tooltip="Simulated Path (OmniDrones)",
        dash_array="5, 5",
    ).add_to(m)

if st.session_state["sim_summary"] is not None:
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st_data = st_folium(
            m, height=650, use_container_width=True,
            returned_objects=["all_drawings", "center", "zoom"],
        )
    with col2:
        st.subheader("Simulation Results")
        
        summary = st.session_state["sim_summary"]
        
        # Display key metrics in a neat layout
        m1, m2 = st.columns(2)
        m1.metric("Flight Time", f"{summary['total_flight_time_sec']} s")
        m2.metric("Distance", f"{summary['total_distance_flown_m']} m")
        
        m3, m4 = st.columns(2)
        m3.metric("Avg Speed", f"{summary['average_speed_m_s']} m/s")
        m4.metric("Battery Used", f"{summary['battery_used_pct']} %")
        
        m5, m6 = st.columns(2)
        m5.metric("Max Deviation", f"{summary['max_deviation_m']} m")
        
        status_color = "normal"
        status_text = "SUCCESS"
        if summary["collision_detected"]:
            status_text = "COLLISION"
            status_color = "inverse"
        m6.metric("Flight Status", status_text, delta="Collision!" if summary["collision_detected"] else None, delta_color=status_color)
        
        st.markdown("---")
        st.markdown("#### Telemetry Charts")
        
        import pandas as pd
        telemetry_df = pd.DataFrame(st.session_state["sim_telemetry"])
        
        # Plot Speed and Battery
        st.caption("Speed Profile (m/s) vs Time (s)")
        st.line_chart(telemetry_df, x="timestamp", y="speed", height=130)
        
        st.caption("Battery Level (%) vs Time (s)")
        st.line_chart(telemetry_df, x="timestamp", y="battery", height=130)
else:
    st_data = st_folium(
        m, height=650, use_container_width=True,
        returned_objects=["all_drawings", "center", "zoom"],
    )

# Persist viewport so the map rebuilds at the same position after every rerun.
if st_data:
    if st_data.get("center"):
        c = st_data["center"]
        st.session_state["map_center"] = [c["lat"], c["lng"]]
    if st_data.get("zoom"):
        st.session_state["map_zoom"] = st_data["zoom"]

# --- Marker capture ---
# Without a stable component key the Draw layer resets on every rerun, so
# all_drawings only ever contains markers placed in the current interaction.
drawings = (st_data or {}).get("all_drawings") or []
new_markers = [d for d in drawings if d.get("geometry", {}).get("type") == "Point"]

if new_markers:
    coord_key = tuple(round(v, 6) for v in new_markers[-1]["geometry"]["coordinates"])
    if coord_key != st.session_state["last_processed_drawing"]:
        st.session_state["last_processed_drawing"] = coord_key
        if not has_start:
            c = new_markers[0]["geometry"]["coordinates"]
            st.session_state["start_point"] = (c[1], c[0])
            st.rerun()
        elif not has_end:
            c = new_markers[0]["geometry"]["coordinates"]
            st.session_state["end_point"] = (c[1], c[0])
            st.rerun()

# --- Calculation Logic ---
if calculate_btn:
    start_pt = st.session_state["start_point"]
    end_pt = st.session_state["end_point"]

    if start_pt is None or end_pt is None:
        st.warning("Place a Start marker and an End marker on the map before calculating.")
        st.stop()

    st.session_state["sim_telemetry"] = None
    st.session_state["sim_summary"] = None

    center_lat = (start_pt[0] + end_pt[0]) / 2.0
    center_lon = (start_pt[1] + end_pt[1]) / 2.0
    lat_diff = abs(start_pt[0] - end_pt[0])
    lon_diff = abs(start_pt[1] - end_pt[1])
    radius_deg = max(lat_diff / 2.0, lon_diff / 2.0) + 0.008
    radius_deg = min(radius_deg, 0.022)  # clamp to keep area under ~20 sq km
    bounds = {
        "north": center_lat + radius_deg,
        "south": center_lat - radius_deg,
        "east":  center_lon + radius_deg,
        "west":  center_lon - radius_deg,
    }

    drone_params = {
        "mass": input_mass,
        "width": input_width,
        "speed": input_speed,
        "alt": input_alt,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
    }

    with st.spinner("Calculating risk footprint..."):
        try:
            risk_map, bounds_list = calculate_risk_area(bounds, drone_params)
        except Exception as exc:
            st.error(f"Risk calculation failed: {exc}")
            st.stop()

    st.session_state["risk_map"] = risk_map
    st.session_state["bounds_list"] = bounds_list
    st.session_state["bounds"] = bounds

    with st.spinner("Computing optimal path..."):
        try:
            path = find_optimal_path(
                risk_map, bounds, start_pt, end_pt, risk_weight
            )
        except Exception as exc:
            st.warning(f"Pathfinding failed: {exc}")
            path = None

    if path is None:
        st.warning("No path found between the two markers.")
        st.session_state["path_latlon"] = None
    else:
        st.session_state["path_latlon"] = path

    st.rerun()

# --- Simulation Logic Execution ---
if st.session_state["path_latlon"] is not None and simulate_btn:
    import requests
    SIM_API_URL = os.getenv("SIM_API_URL", "http://localhost:8000")
    
    # Prepare waypoints
    waypoints = []
    for lat, lon in st.session_state["path_latlon"]:
        waypoints.append({
            "lat": lat,
            "lon": lon,
            "alt": input_alt
        })
        
    drone_params = {
        "mass": input_mass,
        "width": input_width,
        "speed": input_speed,
        "alt": input_alt,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
    }
    
    payload = {
        "waypoints": waypoints,
        "speed": input_speed,
        "drone_params": drone_params
    }
    
    with st.spinner("Running OmniDrones physics simulation..."):
        try:
            response = requests.post(f"{SIM_API_URL}/simulate_path", json=payload, timeout=30)
            if response.status_code == 200:
                res = response.json()
                st.session_state["sim_telemetry"] = res.get("telemetry")
                st.session_state["sim_summary"] = res.get("summary")
                st.toast("Simulation finished!", icon="🚁")
                st.rerun()
            else:
                st.error(f"Simulation API returned error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to simulation API: {e}")
