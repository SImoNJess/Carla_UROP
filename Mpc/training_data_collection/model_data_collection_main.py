from model_data_collection_preparation import *
import numpy as np, pickle, os, sys, datetime, argparse, time

CONTROL_HZ_DEFAULT = 10.0
STEPS_DEFAULT      = 500
PRINT_EVERY        = 50

STEER_RATE_BASE    = 3.00   
STEER_CENTER_BASE  = 6.00  

THR_RATE           = 0.40 
BRK_RATE           = 1.40  
THR_DECAY          = 0.50  
BRK_DECAY          = 0.80  

THR_SCALE_DEFAULT  = 0.70   
IDLE_BRAKE_DEFAULT = 0.0  
MAX_SPEED_DEFAULT  = 0.0   
A_MAX_DEFAULT      = 0.0   

STEER_DEADZONE     = 0.02
THR_DEADZONE       = 0.01
BRK_DEADZONE       = 0.01

LAUNCH_SPEED_MPS   = 1.0   
LAUNCH_THR_CAP     = 0.85   
LAUNCH_STEER_CAP   = 0.90

steer_t = 0.0
thr_t   = 0.0
brk_t   = 0.0

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def current_speed_from_state(st):
    if isinstance(st, (list, tuple, np.ndarray)) and len(st) >= 1:
        for val in reversed(st):
            try:
                v = float(val)
                if 0.0 <= v <= 100.0:
                    return v
            except:
                pass
    return None

class Keys:
    W=0x57; S=0x53; A=0x41; D=0x44
    UP=0x26; DOWN=0x28; LEFT=0x25; RIGHT=0x27
    SPACE=0x20; Q=0x51; ESC=0x1B

class MsvcrtBackend:
    def __init__(self):
        import msvcrt as _m
        self.m = _m
        self.last = set()
    def pump(self):
        m = self.m
        pressed = set()
        while m.kbhit():
            ch = m.getch()
            if ch == b'\x1b':
                pressed.add(Keys.ESC)
            elif ch in (b'\x00', b'\xe0') and m.kbhit():
                ch2 = m.getch()
                table = {b'H':Keys.UP, b'P':Keys.DOWN, b'K':Keys.LEFT, b'M':Keys.RIGHT}
                if ch2 in table: pressed.add(table[ch2])
            else:
                c = ch.lower()
                if c == b'w': pressed.add(Keys.W)
                elif c == b's': pressed.add(Keys.S)
                elif c == b'a': pressed.add(Keys.A)
                elif c == b'd': pressed.add(Keys.D)
                elif c == b' ': pressed.add(Keys.SPACE)
        self.last = pressed
    def key_down(self, vk): return vk in self.last

def _poll_keys_dt(dt, kb, spd=None):
    global steer_t, thr_t, brk_t

    kb.pump()
    left  = kb.key_down(Keys.A) or kb.key_down(Keys.LEFT)
    right = kb.key_down(Keys.D) or kb.key_down(Keys.RIGHT)
    fwd   = kb.key_down(Keys.W) or kb.key_down(Keys.UP)
    bwd   = kb.key_down(Keys.S) or kb.key_down(Keys.DOWN)
    brake_key = kb.key_down(Keys.SPACE)
    esc   = kb.key_down(Keys.Q) or kb.key_down(Keys.ESC)
    if esc: raise SystemExit

    s = float(spd or 0.0)
    fade = max(0.50, 1.0 - (float(spd or 0.0) / 10.0) * 0.50)
    STEER_RATE   = STEER_RATE_BASE   * fade
    STEER_CENTER = STEER_CENTER_BASE * fade

    if left and not right:   steer_t -= STEER_RATE * dt
    elif right and not left: steer_t += STEER_RATE * dt
    else:
        if steer_t > 0: steer_t = max(0.0, steer_t - STEER_CENTER*dt)
        elif steer_t < 0: steer_t = min(0.0, steer_t + STEER_CENTER*dt)
    steer_t = clamp(steer_t, -1.0, 1.0)

    if brake_key:
        brk_t = 1.0; thr_t = 0.0
    elif fwd and not bwd:
        thr_t = min(1.0, thr_t + THR_RATE*dt)
        brk_t = max(0.0, brk_t - BRK_RATE*dt)
    elif bwd and not fwd:
        brk_t = min(1.0, brk_t + BRK_RATE*dt)
        thr_t = max(0.0, thr_t - THR_RATE*dt)
    else:
        thr_t = max(0.0, thr_t - THR_DECAY*THR_RATE*dt)
        brk_t = max(0.0, brk_t - BRK_DECAY*BRK_RATE*dt)

    if s < LAUNCH_SPEED_MPS:
        thr_t  = min(thr_t, LAUNCH_THR_CAP)
        steer_t = clamp(steer_t, -LAUNCH_STEER_CAP, LAUNCH_STEER_CAP)

    if abs(steer_t) < STEER_DEADZONE: steer_t = 0.0
    if thr_t < THR_DEADZONE:  thr_t = 0.0
    if brk_t < BRK_DEADZONE:  brk_t = 0.0

    return steer_t, thr_t, brk_t

def main():
    global CONTROL_HZ, LOG_HZ, CONTROL_DT

    ap = argparse.ArgumentParser()
    ap.add_argument("--control_hz", type=float, default=CONTROL_HZ_DEFAULT, help="Physics/control rate (Hz)")
    ap.add_argument("--steps",      type=int,   default=STEPS_DEFAULT,      help="Rows to collect (exact, 1 per tick)")
    ap.add_argument("--no_render",  action="store_true", default=False,     help="Disable rendering (default False)")
    ap.add_argument("--real_time",  action="store_true", default=True,      help="Pace ticks to wall clock at control_hz (default True)")
    ap.add_argument("--thr_scale",  type=float, default=THR_SCALE_DEFAULT,  help="Throttle scaling [0..1]")
    ap.add_argument("--idle_brake", type=float, default=IDLE_BRAKE_DEFAULT, help="Brake when throttle ≈0")
    ap.add_argument("--max_speed",  type=float, default=MAX_SPEED_DEFAULT,  help="Speed cap (m/s), 0=off")
    ap.add_argument("--a_max",      type=float, default=A_MAX_DEFAULT,      help="Accel cap (m/s^2), 0=off")
    args = ap.parse_args()

    CONTROL_HZ = float(args.control_hz)
    CONTROL_DT = 1.0 / CONTROL_HZ
    LOG_HZ     = CONTROL_HZ  # realtime logging

    print(f"Manual data collection (SYNC world @{CONTROL_HZ:.1f} Hz, 1 row per tick).")
    print("W/Up=throttle, S/Down=brake, A/D or arrows=steer, SPACE=full brake, Q=quit.")
    print(f"Target: {args.steps} rows  ->  ~{args.steps * CONTROL_DT:.1f} s wall time.")
    print("[input] Console keyboard (msvcrt). Make sure THIS window has focus.")

    kb = MsvcrtBackend()
    env = CarEnv()

    try:
        s = env.world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = CONTROL_DT
        if args.no_render:
            s.no_rendering_mode = True
            print("[viz] no_rendering_mode=True")
        else:
            s.no_rendering_mode = False
        env.world.apply_settings(s)
        if hasattr(env, "vehicle") and env.vehicle is not None:
            try: env.vehicle.set_simulate_physics(True)
            except Exception: pass
    except Exception as e:
        print(f"[viz] settings apply failed: {e}")

    systemid_data = []
    prev_state = env.reset()
    sim_time = 0.0

    t0_wall   = time.perf_counter()
    next_wall = t0_wall + CONTROL_DT

    last_spd = current_speed_from_state(prev_state)

    try:
        for i in range(args.steps):
            spd_prev = current_speed_from_state(prev_state)

            steer, thr, brk = _poll_keys_dt(CONTROL_DT, kb, spd=spd_prev)

            thr *= clamp(args.thr_scale, 0.0, 1.0)
            if thr < 1e-3:
                if (spd_prev is not None) and (spd_prev < 0.2):  # m/s threshold (~1.8 km/h)
                    brk = max(brk, args.idle_brake)
                else:
                    brk = 0.0

            if args.max_speed > 0 and (spd_prev is not None) and spd_prev > args.max_speed:
                thr = 0.0
                brk = max(brk, 0.25)

            if args.a_max > 0 and (spd_prev is not None) and (last_spd is not None):
                max_dv = args.a_max * CONTROL_DT
                if (spd_prev - last_spd) > max_dv and thr > 0.0:
                    thr *= 0.5
                    brk = max(brk, 0.2)
            if spd_prev is not None:
                last_spd = spd_prev

            action = (steer, clamp(thr, 0.0, 1.0), clamp(brk, 0.0, 1.0))
            env.apply_action(action)

            env.tick()
            sim_time = (i + 1) * CONTROL_DT

            new_state = env.get_state()
            s_val, t_val, b_val = action
            row = [sim_time] + prev_state + [s_val, t_val, b_val] + new_state
            systemid_data.append(row)
            prev_state = new_state

            if ((i + 1) % PRINT_EVERY) == 0:
                sys.stdout.write(f"\rLogged {i+1}/{args.steps}  (~{CONTROL_HZ:.1f} Hz)")
                sys.stdout.flush()

            if args.real_time:
                now = time.perf_counter()
                lag = now - next_wall
                if lag < 0:
                    time.sleep(-lag)
                next_wall += CONTROL_DT

        wall_elapsed = time.perf_counter() - t0_wall
        print(f"\nFinished {args.steps} rows.  sim={(args.steps*CONTROL_DT):.2f}s, wall={wall_elapsed:.2f}s")

    except SystemExit:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        arr = np.asarray(systemid_data, dtype=float)
        if arr.shape[0] == args.steps:
            os.makedirs("data", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"data/systemid_data_{int(CONTROL_DT*1000)}ms_{ts}.pkl"
            pickle.dump(arr, open(out_path, "wb"))
            print(f"Saved {arr.shape} to {out_path}")
        else:
            print(f"Collected {arr.shape[0]} rows; not saving (target {args.steps}).")
        env.destroy()

if __name__ == "__main__":
    main()
