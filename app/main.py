import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw
from matplotlib import cm
from streamlit_folium import st_folium

from core.engine.interface import calculate_risk_area, find_optimal_path

# --- 1. Page Configuration ---
st.set_page_config(page_title="UAS Ground Risk Analyzer", layout="wide")

# --- 2. Sidebar Inputs ---
st.sidebar.title("Parameters")

st.sidebar.subheader("Visualization")
overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.6)

st.sidebar.markdown("---")
st.sidebar.subheader("Drone Specifications")
drone_preset = st.sidebar.selectbox("Select Preset", ["Custom", "DJI Mavic 3", "DJI Matrice 300 RTK"])

mass, width, speed, alt = 1.0, 0.5, 15.0, 50.0
if drone_preset == "DJI Mavic 3":
    mass, width, speed, alt = 0.9, 0.35, 15.0, 50.0
elif drone_preset == "DJI Matrice 300 RTK":
    mass, width, speed, alt = 9.0, 0.81, 15.0, 100.0

input_mass = st.sidebar.number_input("Mass (kg)", min_value=0.1, max_value=50.0, value=mass)
input_width = st.sidebar.number_input("Width (m)", min_value=0.1, max_value=5.0, value=width)
input_speed = st.sidebar.number_input("Cruise Speed (m/s)", min_value=1.0, max_value=50.0, value=speed)
input_alt = st.sidebar.number_input("Cruise Altitude (m)", min_value=10.0, max_value=500.0, value=alt)

st.sidebar.subheader("Environmental Parameters")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0)
wind_dir = st.sidebar.slider("Wind Direction (degrees)", 0, 360, 90)

st.sidebar.markdown("---")
st.sidebar.subheader("Pathfinding")
risk_weight = st.sidebar.slider("Risk Aversion", 0.0, 100.0, 10.0)
st.sidebar.caption(
    "0 = shortest path only. 100 = strongly avoids high-risk areas. "
    "Place a Start marker then an End marker on the map."
)

# --- 3. Main UI ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("UAS Ground Risk Analyzer")
    st.markdown(
        "1. Draw an operational area polygon. "
        "2. Place a Start marker, then an End marker. "
        "3. Click Calculate."
    )
with col2:
    st.write("")
    st.write("")
    calculate_btn = st.button("Calculate Risk Map", type="primary", use_container_width=True)

# --- 4. Build Map ---
m = folium.Map(location=[51.3360, -0.2670], zoom_start=13)
Draw(
    export=True,
    draw_options={
        "polygon": True,
        "rectangle": True,
        "marker": True,
        "circle": False,
        "circlemarker": False,
        "polyline": False,
    },
).add_to(m)

if "risk_map" in st.session_state and "bounds_list" in st.session_state:
    rm: np.ndarray = st.session_state["risk_map"]
    rm_norm = (rm - rm.min()) / (rm.max() - rm.min() + 1e-9)

    colormap = cm.get_cmap("inferno")
    color_image = colormap(rm_norm)
    color_image[rm_norm < 0.05, 3] = 0.0
    color_image[rm_norm >= 0.05, 3] = 1.0

    folium.raster_layers.ImageOverlay(
        image=color_image,
        bounds=st.session_state["bounds_list"],
        name="Risk Heatmap",
        opacity=overlay_opacity,
    ).add_to(m)
    folium.LayerControl().add_to(m)

if "path_latlon" in st.session_state and st.session_state["path_latlon"]:
    folium.PolyLine(
        locations=st.session_state["path_latlon"],
        color="#00e5ff",
        weight=3,
        opacity=0.9,
        tooltip="Optimal Path",
    ).add_to(m)

st_data = st_folium(m, width=1200, height=600, returned_objects=["all_drawings"])

# --- 5. Calculation Logic ---
if calculate_btn:
    drawings: list[dict] = (st_data or {}).get("all_drawings") or []

    polygons = [
        d for d in drawings
        if d.get("geometry", {}).get("type") in ("Polygon", "Rectangle")
    ]
    markers = [
        d for d in drawings
        if d.get("geometry", {}).get("type") == "Point"
    ]

    if not polygons:
        st.error("Please draw an operational area (rectangle or polygon) on the map first.")
        st.stop()

    geometry = polygons[0]["geometry"]
    coordinates = geometry["coordinates"][0]
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    bounds = {
        "west": min(lons),
        "east": max(lons),
        "south": min(lats),
        "north": max(lats),
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
    st.session_state.pop("path_latlon", None)

    if len(markers) >= 2:
        start_coords = markers[0]["geometry"]["coordinates"]
        end_coords = markers[1]["geometry"]["coordinates"]
        start_latlon = (start_coords[1], start_coords[0])
        end_latlon = (end_coords[1], end_coords[0])

        with st.spinner("Computing optimal path..."):
            try:
                path = find_optimal_path(
                    risk_map, bounds, start_latlon, end_latlon, risk_weight
                )
            except Exception as exc:
                st.warning(f"Pathfinding failed: {exc}")
                path = None

        if path is None:
            st.warning("No path found between the two markers.")
        else:
            st.session_state["path_latlon"] = path
    elif len(markers) == 1:
        st.info("Place a second marker to define the End point for pathfinding.")
    else:
        st.info("Place Start and End markers on the map to compute an optimal path.")

    st.rerun()
