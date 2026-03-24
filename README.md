# UAS Ground Risk Analyzer & Flight Planner

## Overview

A Python-based proof-of-concept tool for generating high-resolution drone ground-strike risk profiles over real urban terrain and computing optimal flight paths that balance shortest distance against minimised population exposure. Risk calculations are grounded in the EASA SORA (Specific Operations Risk Assessment) methodology, making outputs directly relevant to operational authorisation workflows. Designed to demonstrate the feasibility of on-demand, GPU-accelerated risk quantification for urban UAS operations.

---

## Key Features

- **Live building footprint ingestion** via OSMnx, rasterised at 10 m resolution from OpenStreetMap data.
- **GPU-accelerated risk convolution** using a Taichi kernel (Apple Metal on macOS, CUDA on Linux/Windows, CPU fallback).
- **Multi-objective A\* pathfinding** with a tuneable risk-aversion weight, trading off geodesic distance against ground population exposure.

---

## Roadmap

The following capabilities are planned for future iterations:

- **Live meteorological integration** — real-time wind speed and direction ingestion via the Open-Meteo API, replacing manual sidebar inputs.
- **Dynamic battery consumption estimation** — aerodynamic drag modelling as a function of drone mass, cross-section, and cruise speed to produce range-constrained path planning.

---

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| UI & serving | Python 3.12, Streamlit |
| Map rendering | Folium, streamlit-folium |
| GPU compute | Taichi (Metal / CUDA / CPU) |
| Numerical core | NumPy, SciPy |
| Geospatial pipeline | OSMnx, Rasterio, GeoPandas, Shapely, PyProj |
| Testing | Pytest |
| Linting / formatting | Ruff |

---

## Installation

```bash
git clone https://github.com/your-username/drone-risk.git
cd drone-risk

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

Launch the application:

```bash
streamlit run app/main.py
```

**Workflow:**

1. **Configure parameters** — Set drone specifications (mass, width, cruise speed, altitude) and wind conditions in the sidebar. Adjust the risk aversion weight to control the trade-off between path length and population exposure.
2. **Place waypoints** — Drop a **Start** marker then an **End** marker anywhere on the map.
3. **Calculate** — Click **Calculate Risk Map**. The tool fetches live building data, computes the ground-strike risk heatmap, and renders the optimal path overlay.

---

## Testing

```bash
pytest
```

Unit tests cover the A\* pathfinding engine and the core risk pipeline (OSM fetch and Taichi kernel are mocked for offline execution).
