#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod Pod Setup Script — drone-risk simulation API
#
# Run this ONCE inside the RunPod pod via SSH after the pod starts.
# The pod must be using the Isaac Lab 2.3.2 Docker image.
#
# Usage:
#   chmod +x setup_runpod.sh && ./setup_runpod.sh
#
# After this script completes, the FastAPI server will be running on port 8000.
# RunPod will expose it at: https://<pod-id>-8000.proxy.runpod.net
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_URL="https://github.com/srey5/drone-risk.git"   # ← replace with your GitHub URL
REPO_DIR="/workspace/drone-risk"
ISAAC_PYTHON="/workspace/isaaclab/isaaclab.sh"
PORT=8000

echo ""
echo "══════════════════════════════════════════════════════"
echo "  drone-risk · RunPod Simulation API Setup"
echo "══════════════════════════════════════════════════════"

# ── 1. Clone the repository ───────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Repo already cloned — pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/4] Cloning repo to $REPO_DIR ..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ── 2. Install FastAPI deps into Isaac Lab's Python ───────────────────────────
echo "[2/4] Installing FastAPI + Uvicorn into Isaac Lab Python environment..."
"$ISAAC_PYTHON" -p -m pip install --no-cache-dir fastapi>=0.100.0 uvicorn>=0.22.0 pydantic>=2.0.0

# ── 3. Verify Isaac Lab can import omni modules ───────────────────────────────
echo "[3/4] Verifying Isaac Lab omni imports..."
"$ISAAC_PYTHON" -p -c "from isaaclab.app import AppLauncher; print('  ✓ isaaclab imports OK')"

# ── 4. Start the FastAPI server ───────────────────────────────────────────────
echo "[4/4] Starting FastAPI server on port $PORT..."
echo ""
echo "  Server URL (RunPod proxy):"
echo "  https://$(hostname)-${PORT}.proxy.runpod.net"
echo ""
echo "  Ctrl+C to stop. Re-run this script to restart."
echo ""

PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
    "$ISAAC_PYTHON" -p -m uvicorn sim_api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
