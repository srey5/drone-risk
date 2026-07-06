# ── Isaac Lab 2.3.2 on Isaac Sim (requires NVIDIA RTX GPU with RT Cores)
# ⚠️  Do NOT run on A100/H100 — they lack RT Cores and Isaac Sim will fail to start.
# Recommended RunPod GPU: RTX 4090 (24 GB) or RTX A6000 (48 GB)
FROM nvcr.io/nvidia/isaac-lab:2.3.2

# ── Environment ──────────────────────────────────────────────────────────────
ENV ACCEPT_EULA=Y
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
# Put the project root on the Python path so `from sim_api.omni_runner` resolves
ENV PYTHONPATH=/workspace/drone-risk:${PYTHONPATH}

# ── Copy source ───────────────────────────────────────────────────────────────
WORKDIR /workspace/drone-risk
COPY . /workspace/drone-risk/sim_api/

# ── Install our extra deps into Isaac Lab's bundled Python environment ────────
# isaaclab.sh -p  = run with Isaac Lab's own Python (has omni.* already available)
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.22.0 \
    pydantic>=2.0.0

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the server ──────────────────────────────────────────────────────────
# Module path is sim_api.main:app because WORKDIR=/workspace/drone-risk
CMD ["/workspace/isaaclab/isaaclab.sh", "-p", "-m", \
     "uvicorn", "sim_api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
