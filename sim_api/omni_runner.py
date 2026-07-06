import math
import random
from typing import List, Dict, Any, Tuple

# Fallback state if Isaac Lab is not installed (e.g. running locally on macOS)
HAS_ISAAC = False
try:
    # Launch simulation app first before importing other omni modules
    # pyrefly: ignore [missing-import]
    from isaaclab.app import AppLauncher

    # Setup AppLauncher in headless mode
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import isaaclab.sim as sim_utils
    # pyrefly: ignore [missing-import]
    from isaaclab.sim import SimulationContext
    # pyrefly: ignore [missing-import]
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    HAS_ISAAC = True
except ImportError as e:
    print(f"Isaac Lab not found. Running in mock simulator mode. Error: {e}")

# --- ENU / GPS Coordinate Conversion Math ---
def gps_to_local(lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float) -> Tuple[float, float, float]:
    """
    Convert GPS (lat, lon, alt) to local tangent plane Cartesian coordinates (x, y, z)
    using a flat-earth approximation (ENU: East, North, Up).
    """
    lat_rad = math.radians(ref_lat)
    y = (lat - ref_lat) * 111111.0
    x = (lon - ref_lon) * 111111.0 * math.cos(lat_rad)
    z = alt
    return x, y, z

def local_to_gps(x: float, y: float, z: float, ref_lat: float, ref_lon: float) -> Tuple[float, float, float]:
    """
    Convert local ENU coordinates (x, y, z) back to GPS (lat, lon, alt).
    """
    lat_rad = math.radians(ref_lat)
    lat = ref_lat + (y / 111111.0)
    lon = ref_lon + (x / (111111.0 * math.cos(lat_rad)))
    alt = z
    return lat, lon, alt

# --- Initialize Isaac Lab Context at Startup if Available ---
if HAS_ISAAC:
    print("Initializing Isaac Lab Simulation Context...")
    # 200Hz physics step (dt = 0.005)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)

    # Spawn ground plane
    gp_cfg = sim_utils.GroundPlaneCfg()
    gp_cfg.func("/World/defaultGroundPlane", gp_cfg)

    # Configure Crazyflie drone asset
    crazyflie_cfg = RigidObjectCfg(
        prim_path="/World/Crazyflie",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                kinematic_enabled=False,
                disable_gravity=True,  # Disable gravity to run kinematic-dynamics overlay control
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0)
        )
    )

    # Instantiate RigidObject
    crazyflie = RigidObject(crazyflie_cfg)
    sim.reset()
    print("Isaac Lab Environment initialized successfully.")

# --- Real Isaac Lab Simulator ---
def simulate_flight_isaac(
    waypoints: List[Dict[str, float]], 
    speed: float, 
    drone_params: Dict[str, Any]
) -> Dict[str, Any]:
    ref_lat = waypoints[0]["lat"]
    ref_lon = waypoints[0]["lon"]
    
    # 1. Convert GPS waypoints to local coordinate system (ENU)
    local_wps = []
    for wp in waypoints:
        x, y, z = gps_to_local(wp["lat"], wp["lon"], wp.get("alt", drone_params.get("alt", 50.0)), ref_lat, ref_lon)
        local_wps.append((x, y, z))
        
    # Get drone properties
    mass = drone_params.get("mass", 1.0)
    width = drone_params.get("width", 0.5)
    wind_speed = drone_params.get("wind_speed", 0.0)
    wind_dir_deg = drone_params.get("wind_dir", 0.0)
    
    # Convert wind direction to radians
    wind_rad = math.radians(90 - wind_dir_deg)
    wind_dx = wind_speed * math.cos(wind_rad)
    wind_dy = wind_speed * math.sin(wind_rad)
    
    # Start position
    start_x, start_y, start_z = local_wps[0]
    
    # Reset/Teleport drone to start position in Isaac Sim
    pos_tensor = torch.tensor([[start_x, start_y, start_z, 1.0, 0.0, 0.0, 0.0]], device=crazyflie.device)
    vel_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=crazyflie.device)
    crazyflie.write_root_pose_to_sim(pos_tensor)
    crazyflie.write_root_velocity_to_sim(vel_tensor)
    crazyflie.reset()
    
    telemetry = []
    current_time = 0.0
    dt = 0.005  # 200Hz physics step
    sampling_interval = 0.1  # 10Hz telemetry logging
    last_log_time = -sampling_interval
    
    curr_x, curr_y, curr_z = start_x, start_y, start_z
    curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0
    
    battery = 100.0
    total_energy_capacity = 3500.0
    power_draw_hover = mass * 9.81 * 12.0
    drag_coeff = 0.15 * width * mass
    
    wp_idx = 1
    num_wps = len(local_wps)
    max_deviation = 0.0
    total_distance_flown = 0.0
    
    max_steps = 15000 * 20  # 200Hz max steps safety limit
    step = 0
    
    while wp_idx < num_wps and step < max_steps:
        step += 1
        target_x, target_y, target_z = local_wps[wp_idx]
        
        # Vector to target
        dx = target_x - curr_x
        dy = target_y - curr_y
        dz = target_z - curr_z
        dist_to_target = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist_to_target < 2.0:
            wp_idx += 1
            continue
            
        # P-controller for velocity command
        k_p = 0.8
        desired_vx = (dx / dist_to_target) * speed
        desired_vy = (dy / dist_to_target) * speed
        desired_vz = (dz / dist_to_target) * speed
        
        # Smooth deceleration near final waypoint
        if wp_idx == num_wps - 1 and dist_to_target < 10.0:
            speed_factor = max(0.15, dist_to_target / 10.0)
            desired_vx *= speed_factor
            desired_vy *= speed_factor
            desired_vz *= speed_factor
            
        # Wind drag force scaling
        wind_fx = drag_coeff * (wind_dx - curr_vx)
        wind_fy = drag_coeff * (wind_dy - curr_vy)
        
        # Acceleration commands
        ax = (desired_vx - curr_vx) * k_p + (wind_fx / mass)
        ay = (desired_vy - curr_vy) * k_p + (wind_fy / mass)
        az = (desired_vz - curr_vz) * k_p
        
        # Clamp accelerations
        max_accel_horiz = 3.0
        max_accel_vert = 4.0
        accel_h = math.sqrt(ax*ax + ay*ay)
        if accel_h > max_accel_horiz:
            ax = (ax / accel_h) * max_accel_horiz
            ay = (ay / accel_h) * max_accel_horiz
        if abs(az) > max_accel_vert:
            az = math.copysign(max_accel_vert, az)
            
        # Update velocities and positions
        curr_vx += ax * dt
        curr_vy += ay * dt
        curr_vz += az * dt
        
        prev_x, prev_y, prev_z = curr_x, curr_y, curr_z
        curr_x += curr_vx * dt
        curr_y += curr_vy * dt
        curr_z += curr_vz * dt
        
        step_dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)
        total_distance_flown += step_dist
        
        # Pitch, Roll, Yaw Euler angles estimation
        heading = math.atan2(curr_vy, curr_vx) if (curr_vx**2 + curr_vy**2) > 0.01 else 0.0
        yaw = math.degrees(heading)
        accel_body_x = ax * math.cos(heading) + ay * math.sin(heading)
        accel_body_y = -ax * math.sin(heading) + ay * math.cos(heading)
        pitch = math.degrees(math.atan2(accel_body_x, 9.81))
        roll = math.degrees(math.atan2(-accel_body_y, 9.81))
        
        # Convert orientation to quaternion (w, x, y, z)
        cy = math.cos(math.radians(yaw) * 0.5)
        sy = math.sin(math.radians(yaw) * 0.5)
        cp = math.cos(math.radians(pitch) * 0.5)
        sp = math.sin(math.radians(pitch) * 0.5)
        cr = math.cos(math.radians(roll) * 0.5)
        sr = math.sin(math.radians(roll) * 0.5)
        
        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy
        
        # Write state changes into Isaac Lab Simulator
        pos_tensor = torch.tensor([[curr_x, curr_y, curr_z, q_w, q_x, q_y, q_z]], device=crazyflie.device)
        vel_tensor = torch.tensor([[curr_vx, curr_vy, curr_vz, 0.0, 0.0, 0.0]], device=crazyflie.device)
        
        crazyflie.write_root_pose_to_sim(pos_tensor)
        crazyflie.write_root_velocity_to_sim(vel_tensor)
        
        # Step simulation physics context
        sim.step()
        
        # Retrieve computed simulation updates (resolves PhysX interactions)
        sim_pos = crazyflie.data.root_pos_w[0].cpu().numpy()
        sim_vel = crazyflie.data.root_vel_w[0, 0:3].cpu().numpy()
        
        curr_x, curr_y, curr_z = sim_pos[0], sim_pos[1], sim_pos[2]
        curr_vx, curr_vy, curr_vz = sim_vel[0], sim_vel[1], sim_vel[2]
        
        # Track cross track error
        p_start = local_wps[wp_idx-1]
        p_end = local_wps[wp_idx]
        seg_dx = p_end[0] - p_start[0]
        seg_dy = p_end[1] - p_start[1]
        seg_dz = p_end[2] - p_start[2]
        seg_len = math.sqrt(seg_dx**2 + seg_dy**2 + seg_dz**2)
        if seg_len > 1e-3:
            curr_dx = curr_x - p_start[0]
            curr_dy = curr_y - p_start[1]
            curr_dz = curr_z - p_start[2]
            proj = (curr_dx * seg_dx + curr_dy * seg_dy + curr_dz * seg_dz) / (seg_len * seg_len)
            proj = max(0.0, min(1.0, proj))
            nearest_x = p_start[0] + proj * seg_dx
            nearest_y = p_start[1] + proj * seg_dy
            nearest_z = p_start[2] + proj * seg_dz
            deviation = math.sqrt((curr_x - nearest_x)**2 + (curr_y - nearest_y)**2 + (curr_z - nearest_z)**2)
            max_deviation = max(max_deviation, deviation)
            
        # Battery depletion calculations
        curr_speed = math.sqrt(curr_vx**2 + curr_vy**2 + curr_vz**2)
        power_aero = drag_coeff * (curr_speed ** 3)
        power_climb = mass * 9.81 * max(0.0, curr_vz)
        total_power = power_draw_hover + power_aero + power_climb
        energy_used = (total_power * dt) / 3600.0
        battery_drain = (energy_used / total_energy_capacity) * 100.0
        battery = max(0.0, battery - battery_drain)
        
        current_time += dt
        
        # Log telemetry at 10Hz sampling rate
        if current_time - last_log_time >= sampling_interval:
            last_log_time = current_time
            curr_lat, curr_lon, curr_alt = local_to_gps(curr_x, curr_y, curr_z, ref_lat, ref_lon)
            telemetry.append({
                "timestamp": round(current_time, 2),
                "lat": round(curr_lat, 6),
                "lon": round(curr_lon, 6),
                "alt": round(curr_alt, 2),
                "x": round(curr_x, 2),
                "y": round(curr_y, 2),
                "z": round(curr_z, 2),
                "vx": round(curr_vx, 2),
                "vy": round(curr_vy, 2),
                "vz": round(curr_vz, 2),
                "speed": round(curr_speed, 2),
                "battery": round(battery, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
                "yaw": round(yaw, 2),
                "wind_force_x": round(wind_fx, 3),
                "wind_force_y": round(wind_fy, 3)
            })
            
    total_flight_time = current_time
    battery_consumed = 100.0 - battery
    avg_speed = total_distance_flown / total_flight_time if total_flight_time > 0 else 0.0
    
    # Basic collision detector fallback
    collision_detected = False
    for t in telemetry:
        if t["alt"] < 1.0 and t["timestamp"] > 2.0:
            collision_detected = True
            break
            
    return {
        "status": "completed",
        "telemetry": telemetry,
        "summary": {
            "total_flight_time_sec": round(total_flight_time, 1),
            "total_distance_flown_m": round(total_distance_flown, 1),
            "average_speed_m_s": round(avg_speed, 2),
            "battery_used_pct": round(battery_consumed, 2),
            "max_deviation_m": round(max_deviation, 2),
            "collision_detected": collision_detected
        }
    }

# --- Pure Python Fallback Mock Simulator ---
def simulate_flight_mock(
    waypoints: List[Dict[str, float]], 
    speed: float, 
    drone_params: Dict[str, Any]
) -> Dict[str, Any]:
    ref_lat = waypoints[0]["lat"]
    ref_lon = waypoints[0]["lon"]
    
    local_wps = []
    for wp in waypoints:
        x, y, z = gps_to_local(wp["lat"], wp["lon"], wp.get("alt", drone_params.get("alt", 50.0)), ref_lat, ref_lon)
        local_wps.append((x, y, z))
        
    mass = drone_params.get("mass", 1.0)
    width = drone_params.get("width", 0.5)
    wind_speed = drone_params.get("wind_speed", 0.0)
    wind_dir_deg = drone_params.get("wind_dir", 0.0)
    
    wind_rad = math.radians(90 - wind_dir_deg)
    wind_dx = wind_speed * math.cos(wind_rad)
    wind_dy = wind_speed * math.sin(wind_rad)
    
    telemetry = []
    current_time = 0.0
    dt = 0.1 # 10Hz sampling
    
    curr_x, curr_y, curr_z = local_wps[0]
    curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0
    
    battery = 100.0
    total_energy_capacity = 3500.0
    power_draw_hover = mass * 9.81 * 12.0
    
    wp_idx = 1
    num_wps = len(local_wps)
    max_deviation = 0.0
    total_distance_flown = 0.0
    
    curr_lat, curr_lon, curr_alt = local_to_gps(curr_x, curr_y, curr_z, ref_lat, ref_lon)
    telemetry.append({
        "timestamp": current_time,
        "lat": curr_lat,
        "lon": curr_lon,
        "alt": curr_alt,
        "x": curr_x,
        "y": curr_y,
        "z": curr_z,
        "vx": curr_vx,
        "vy": curr_vy,
        "vz": curr_vz,
        "speed": 0.0,
        "battery": battery,
        "pitch": 0.0,
        "roll": 0.0,
        "yaw": 0.0,
        "wind_force_x": 0.0,
        "wind_force_y": 0.0
    })
    
    max_steps = 15000
    step = 0
    
    while wp_idx < num_wps and step < max_steps:
        step += 1
        target_x, target_y, target_z = local_wps[wp_idx]
        
        dx = target_x - curr_x
        dy = target_y - curr_y
        dz = target_z - curr_z
        dist_to_target = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist_to_target < 2.0:
            wp_idx += 1
            continue
            
        k_p = 0.8
        desired_vx = (dx / dist_to_target) * speed
        desired_vy = (dy / dist_to_target) * speed
        desired_vz = (dz / dist_to_target) * speed
        
        if wp_idx == num_wps - 1 and dist_to_target < 10.0:
            speed_factor = max(0.15, dist_to_target / 10.0)
            desired_vx *= speed_factor
            desired_vy *= speed_factor
            desired_vz *= speed_factor
            
        drag_coeff = 0.15 * width * mass
        wind_fx = drag_coeff * (wind_dx - curr_vx)
        wind_fy = drag_coeff * (wind_dy - curr_vy)
        
        ax = (desired_vx - curr_vx) * k_p + (wind_fx / mass)
        ay = (desired_vy - curr_vy) * k_p + (wind_fy / mass)
        az = (desired_vz - curr_vz) * k_p
        
        max_accel_horiz = 3.0
        max_accel_vert = 4.0
        accel_h = math.sqrt(ax*ax + ay*ay)
        if accel_h > max_accel_horiz:
            ax = (ax / accel_h) * max_accel_horiz
            ay = (ay / accel_h) * max_accel_horiz
        if abs(az) > max_accel_vert:
            az = math.copysign(max_accel_vert, az)
            
        curr_vx += ax * dt
        curr_vy += ay * dt
        curr_vz += az * dt
        
        prev_x, prev_y, prev_z = curr_x, curr_y, curr_z
        curr_x += curr_vx * dt
        curr_y += curr_vy * dt
        curr_z += curr_vz * dt
        
        step_dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)
        total_distance_flown += step_dist
        
        p_start = local_wps[wp_idx-1]
        p_end = local_wps[wp_idx]
        seg_dx = p_end[0] - p_start[0]
        seg_dy = p_end[1] - p_start[1]
        seg_dz = p_end[2] - p_start[2]
        seg_len = math.sqrt(seg_dx**2 + seg_dy**2 + seg_dz**2)
        if seg_len > 1e-3:
            curr_dx = curr_x - p_start[0]
            curr_dy = curr_y - p_start[1]
            curr_dz = curr_z - p_start[2]
            proj = (curr_dx * seg_dx + curr_dy * seg_dy + curr_dz * seg_dz) / (seg_len * seg_len)
            proj = max(0.0, min(1.0, proj))
            nearest_x = p_start[0] + proj * seg_dx
            nearest_y = p_start[1] + proj * seg_dy
            nearest_z = p_start[2] + proj * seg_dz
            deviation = math.sqrt((curr_x - nearest_x)**2 + (curr_y - nearest_y)**2 + (curr_z - nearest_z)**2)
            max_deviation = max(max_deviation, deviation)
            
        heading = math.atan2(curr_vy, curr_vx) if (curr_vx**2 + curr_vy**2) > 0.01 else 0.0
        yaw = math.degrees(heading)
        accel_body_x = ax * math.cos(heading) + ay * math.sin(heading)
        accel_body_y = -ax * math.sin(heading) + ay * math.cos(heading)
        pitch = math.degrees(math.atan2(accel_body_x, 9.81))
        roll = math.degrees(math.atan2(-accel_body_y, 9.81))
        
        pitch += random.normalvariate(0.0, 0.5)
        roll += random.normalvariate(0.0, 0.5)
        yaw += random.normalvariate(0.0, 0.2)
        
        curr_speed = math.sqrt(curr_vx**2 + curr_vy**2 + curr_vz**2)
        power_aero = drag_coeff * (curr_speed ** 3)
        power_climb = mass * 9.81 * max(0.0, curr_vz)
        total_power = power_draw_hover + power_aero + power_climb
        energy_used = (total_power * dt) / 3600.0
        battery_drain = (energy_used / total_energy_capacity) * 100.0
        battery = max(0.0, battery - battery_drain)
        
        current_time += dt
        curr_lat, curr_lon, curr_alt = local_to_gps(curr_x, curr_y, curr_z, ref_lat, ref_lon)
        
        telemetry.append({
            "timestamp": round(current_time, 2),
            "lat": round(curr_lat, 6),
            "lon": round(curr_lon, 6),
            "alt": round(curr_alt, 2),
            "x": round(curr_x, 2),
            "y": round(curr_y, 2),
            "z": round(curr_z, 2),
            "vx": round(curr_vx, 2),
            "vy": round(curr_vy, 2),
            "vz": round(curr_vz, 2),
            "speed": round(curr_speed, 2),
            "battery": round(battery, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "yaw": round(yaw, 2),
            "wind_force_x": round(wind_fx, 3),
            "wind_force_y": round(wind_fy, 3)
        })
        
    total_flight_time = current_time
    battery_consumed = 100.0 - battery
    avg_speed = total_distance_flown / total_flight_time if total_flight_time > 0 else 0.0
    
    collision_detected = False
    for t in telemetry:
        if t["alt"] < 1.0 and t["timestamp"] > 2.0:
            collision_detected = True
            break
            
    return {
        "status": "completed",
        "telemetry": telemetry,
        "summary": {
            "total_flight_time_sec": round(total_flight_time, 1),
            "total_distance_flown_m": round(total_distance_flown, 1),
            "average_speed_m_s": round(avg_speed, 2),
            "battery_used_pct": round(battery_consumed, 2),
            "max_deviation_m": round(max_deviation, 2),
            "collision_detected": collision_detected
        }
    }

# --- Main Unified Entrypoint ---
def simulate_flight(
    waypoints: List[Dict[str, float]], 
    speed: float, 
    drone_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate a flight trajectory along the provided GPS waypoints.
    If NVIDIA Isaac Lab is available (on RunPod GPU), runs the real PhysX physics loop.
    Otherwise, gracefully falls back to the pure Python mathematical flight dynamics model.
    """
    if HAS_ISAAC:
        try:
            return simulate_flight_isaac(waypoints, speed, drone_params)
        except Exception as e:
            print(f"Isaac Lab simulation runtime error: {e}. Falling back to mock model.")
            return simulate_flight_mock(waypoints, speed, drone_params)
    else:
        return simulate_flight_mock(waypoints, speed, drone_params)
