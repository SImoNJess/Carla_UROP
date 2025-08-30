# mpc_replay.py
import os, json, math, time, argparse, numpy as np
import carla

REPLAY_DT  = None
USE_PHYSICS = False
DRAW_TRAJ   = True
FOLLOW_CAM  = True

def yaw_rad_to_deg(yaw_rad): 
    return math.degrees(yaw_rad)

def main(plan_path):
    client = carla.Client("localhost", 2000)
    client.set_timeout(6.0)

    data = np.load(plan_path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    town = meta["town"]
    dt   = float(meta["dt"])
    global REPLAY_DT; REPLAY_DT = dt
    spawn_x, spawn_y, spawn_z, spawn_yaw = meta["spawn_xyz_yaw"]

    world = client.load_world(town)
    world.apply_settings(carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=dt
    ))

    m = world.get_map()
    blueprints = world.get_blueprint_library()
    bp = blueprints.find("vehicle.tesla.model3")
    ego = world.try_spawn_actor(bp, carla.Transform(
        carla.Location(spawn_x, spawn_y, spawn_z),
        carla.Rotation(yaw=spawn_yaw)
    ))
    if ego is None:
        raise RuntimeError("Failed to spawn ego. Try a different spawn or cleanup leftover actors.")

    t   = data["t"]
    xs  = data["x"]; ys = data["y"]; vs = data["v"]
    if "phi" in data.files:
        yaws = data["phi"]
    elif "yaw" in data.files:
        yaws = data["yaw"]
    else:
        raise RuntimeError("Yaw array not found in plan file.")

    steer = data.get("steer", None)
    thr   = data.get("throttle", None)
    brk   = data.get("brake", None)

    # draw planned route same as second file (red line through (x,y))
    if DRAW_TRAJ:
        step = max(1, len(xs)//300)
        prev = None
        for i in range(0, len(xs), step):
            loc = m.get_waypoint(carla.Location(float(xs[i]), float(ys[i]), 0.0)).transform.location
            if prev is not None:
                world.debug.draw_line(prev, loc, thickness=0.05, life_time=0.0, color=carla.Color(255,0,0))
            prev = loc
        if prev is not None:
            world.debug.draw_string(prev, "END", life_time=0.0, color=carla.Color(255,0,0))

    if not USE_PHYSICS:
        ego.set_simulate_physics(False)

    spectator = world.get_spectator()
    def follow(x, y, yaw_deg):
        base_z = m.get_waypoint(carla.Location(x, y, 0)).transform.location.z + 8.0
        loc = carla.Location(x, y, base_z)
        back = carla.Location(x=-15*math.cos(math.radians(yaw_deg)),
                              y=-15*math.sin(math.radians(yaw_deg)), z=0)
        spectator.set_transform(carla.Transform(loc + back, carla.Rotation(pitch=-25, yaw=yaw_deg)))

    for k in range(len(t)):
        yaw_deg = yaw_rad_to_deg(float(yaws[k]))
        z = m.get_waypoint(carla.Location(float(xs[k]), float(ys[k]), 0)).transform.location.z
        if not USE_PHYSICS:
            ego.set_transform(carla.Transform(
                carla.Location(float(xs[k]), float(ys[k]), z),
                carla.Rotation(yaw=yaw_deg)
            ))
        else:
            if steer is None or thr is None or brk is None:
                raise RuntimeError("Control arrays missing in plan file.")
            ego.apply_control(carla.VehicleControl(
                throttle=float(thr[k]) if k < len(thr) else 0.0,
                steer=float(steer[k]) if k < len(steer) else 0.0,
                brake=float(brk[k]) if k < len(brk) else 0.0
            ))

        if FOLLOW_CAM:
            follow(float(xs[k]), float(ys[k]), yaw_deg)

        world.tick()

    print("[OK] Replay complete.")
    time.sleep(0.5)
    ego.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Path to mpc_plan_*.npz")
    args = parser.parse_args()
    main(args.plan)
