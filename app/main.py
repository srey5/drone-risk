import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib
from folium.plugins import Draw

from core.engine.interface import (
    bounds_from_points,
    calculate_risk_area,
    find_optimal_path,
)

# --- Page Configuration ---
st.set_page_config(page_title="UAS Ground Risk Analyzer", layout="wide")

# --- State Initialization ---
for key in ("start_point", "end_point", "risk_map", "bounds_list", "bounds", "path_latlon"):
    if key not in st.session_state:
        st.session_state[key] = None

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
if st.sidebar.button("Reset Points"):
    for key in ("start_point", "end_point", "risk_map", "bounds_list", "bounds", "path_latlon"):
        st.session_state[key] = None
    st.rerun()

# --- Build Map ---
has_start = st.session_state["start_point"] is not None
has_end = st.session_state["end_point"] is not None

if has_start and has_end:
    st.caption("Start and End points set. Adjust parameters and click Calculate.")
elif has_start:
    st.caption("Start point set. Drop an End marker on the map.")
else:
    st.caption("Drop a Start marker and an End marker, then click Calculate.")

m = folium.Map(location=[51.3360, -0.2670], zoom_start=13)
Draw(
    export=False,
    draw_options={
        "polygon": False,
        "rectangle": False,
        "marker": True,
        "circle": False,
        "circlemarker": False,
        "polyline": False,
    },
    edit_options={"remove": True},
).add_to(m)

# Re-inject persisted waypoints onto the map
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
    color_image[:, :, 3] = rm_norm ** 0.5  # smooth fade: high risk opaque, low risk transparent
    color_image[rm_norm < 0.05, 3] = 0.0   # hide near-zero noise entirely

    folium.raster_layers.ImageOverlay(
        image=color_image,
        bounds=st.session_state["bounds_list"],
        name="Risk Heatmap",
        opacity=overlay_opacity,
    ).add_to(m)
    folium.LayerControl().add_to(m)

# Path overlay
if st.session_state["path_latlon"] is not None:
    path_coords = st.session_state["path_latlon"]
    folium.PolyLine(
        locations=path_coords,
        color="#00e5ff",
        weight=3,
        opacity=0.9,
        tooltip="Optimal Path",
    ).add_to(m)

st_data = st_folium(
    m, height=700, use_container_width=True, returned_objects=["all_drawings"]
)

# --- Capture new markers from the draw layer ---
drawings = (st_data or {}).get("all_drawings") or []
new_markers = [
    d for d in drawings
    if d.get("geometry", {}).get("type") == "Point"
]

if new_markers:
    # Assign markers in order: first new marker fills start, second fills end
    if not has_start:
        c = new_markers[0]["geometry"]["coordinates"]
        st.session_state["start_point"] = (c[1], c[0])
        if len(new_markers) >= 2:
            c2 = new_markers[1]["geometry"]["coordinates"]
            st.session_state["end_point"] = (c2[1], c2[0])
        st.rerun()
    elif not has_end:
        c = new_markers[0]["geometry"]["coordinates"]
        st.session_state["end_point"] = (c[1], c[0])
        st.rerun()
    else:
        # Both already set; latest marker replaces end point
        c = new_markers[-1]["geometry"]["coordinates"]
        st.session_state["end_point"] = (c[1], c[0])
        st.rerun()

# --- Calculation Logic ---
if calculate_btn:
    start_pt = st.session_state["start_point"]
    end_pt = st.session_state["end_point"]

    if start_pt is None or end_pt is None:
        st.warning("Place exactly two markers on the map: a Start point and an End point.")
        st.stop()

    bounds = bounds_from_points(start_pt, end_pt, padding=0.2)

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
