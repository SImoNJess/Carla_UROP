import os, sys, time, math, json
import numpy as np
import carla

from ilqr_numpy_blackbox import BlackBoxDynamics, CostFunction, ilqr_optimize

DT              = 0.1
H               = 50
APPLY_STEPS     = 3
TOTAL_SECONDS   = 20.0
SPEED_REF       = 8.0
ROUTE_AHEAD_M   = 250.0
STEP_M          = 2.0
TOWN            = "Town04_Opt"

VERBOSE         = True
SAVE_DIR        = "plans"
PARTIAL_EVERY   = 20

os.makedirs(SAVE_DIR, exist_ok=True)
PARTIAL_STAMP = time.strftime("%Y%m%d_%H%M%S")

def forward_centerline(map_, start_wp, ahead_m=ROUTE_AHEAD_M, step_m=STEP_M):
    wp = start_wp
    pts = [(wp.transform.location.x, wp.transform.location.y)]
    acc = 0.0
    while acc < ahead_m:
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        loc = wp.transform.location
        pts.append((loc.x, loc.y))
        acc += step_m
    return np.asarray(pts, dtype=np.float32)

def trim_waypoints(waypoints, x, y, keep_radius=3.0):
    if waypoints is None or len(waypoints) == 0:
        return waypoints
    dx = waypoints[:, 0] - x
    dy = waypoints[:, 1] - y
    dist2 = dx * dx + dy * dy
    keep_idx = np.where(dist2 > keep_radius * keep_radius)[0]
    if keep_idx.size == 0:
        return waypoints[-10:]
    return waypoints[keep_idx[0]:]

def save_npz(path, t_log, X_log, U_log, meta):
    X = np.asarray(X_log, np.float32)
    U = np.asarray(U_log, np.float32) if U_log else np.zeros((0, 3), np.float32)
    T = np.asarray(t_log, np.float32)
    np.savez_compressed(
        path,
        t=T,
        x=X[:, 0], y=X[:, 1], v=X[:, 2], phi=X[:, 3], beta=X[:, 4],
        steer=U[:, 0] if U.shape[0] else np.zeros((0,), np.float32),
        throttle=U[:, 1] if U.shape[0] else np.zeros((0,), np.float32),
        brake=U[:, 2] if U.shape[0] else np.zeros((0,), np.float32),
        meta=json.dumps(meta),
    )

def save_partial(t_log, X_log, U_log, meta):
    path = os.path.join(SAVE_DIR, f"mpc_plan_partial_{PARTIAL_STAMP}.npz")
    save_npz(path, t_log, X_log, U_log, meta)
    print(f"[PARTIAL] saved {len(t_log)} steps -> {path}", flush=True)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.load_world(TOWN)
    world.apply_settings(carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=DT,
        no_rendering_mode=True
    ))

    m = world.get_map()
    spawn = m.get_spawn_points()[0]
    start_wp = m.get_waypoint(spawn.location, project_to_road=True)

    ref_xy0 = forward_centerline(m, start_wp)

    phi0 = math.radians(spawn.rotation.yaw)
    x_nom = np.array([spawn.location.x, spawn.location.y, 0.0, phi0, 0.0], dtype=np.float32)

    dyn = BlackBoxDynamics(dt=DT)

    T_steps = int(TOTAL_SECONDS / DT)
    X_log = [x_nom.copy()]
    U_log = []
    t_log = [0.0]

    meta = dict(
        town=TOWN,
        dt=DT,
        spawn_xyz_yaw=[spawn.location.x, spawn.location.y, spawn.location.z, spawn.rotation.yaw],
        route_centerline_xy=ref_xy0.tolist(),
    )

    k = 0
    start_all = time.perf_counter()
    ref_xy = ref_xy0.copy()

    try:
        while k < T_steps:
            loop_t0 = time.perf_counter()

            ref_xy = trim_waypoints(ref_xy, X_log[-1][0], X_log[-1][1])
            if ref_xy is None or len(ref_xy) < 5:
                cur_wp = m.get_waypoint(
                    carla.Location(X_log[-1][0], X_log[-1][1], 0.0),
                    project_to_road=True
                )
                ref_xy = forward_centerline(m, cur_wp)

            cost = CostFunction(ref_xy, dp=1.0, speed_ref=SPEED_REF)

            u_init = np.zeros((H, 3), dtype=np.float32)
            u_init[:, 1] = 0.2

            t0 = time.perf_counter()
            x_trj, u_trj = ilqr_optimize(dyn, cost, X_log[-1], u_init, max_iter=40)
            t1 = time.perf_counter()

            steps = min(APPLY_STEPS, H, T_steps - k)
            for j in range(steps):
                U_log.append(u_trj[j])
                X_log.append(x_trj[j + 1])
                t_log.append(t_log[-1] + DT)
            k += steps

            if VERBOSE:
                done_pct = 100.0 * k / T_steps
                print(
                    f"[loading...] {k:4d}/{T_steps} steps ({done_pct:5.1f}%) | "
                    f"iLQR {t1 - t0:.2f}s | loop {time.perf_counter() - loop_t0:.2f}s | "
                    f"elapsed {time.perf_counter() - start_all:.1f}s",
                    flush=True
                )

            if (k % PARTIAL_EVERY) == 0:
                save_partial(t_log, X_log, U_log, meta)

    except KeyboardInterrupt:
        print("\n[CTRL-C] stopping; writing partial...", flush=True)
        save_partial(t_log, X_log, U_log, meta)
        sys.exit(0)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SAVE_DIR, f"mpc_plan_{ts}.npz")
    save_npz(out_path, t_log, X_log, U_log, meta)
    print(f"[0v0] saved full plan -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
