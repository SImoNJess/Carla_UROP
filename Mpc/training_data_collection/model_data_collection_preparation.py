import time, random, math
from enum import Enum

import carla
try:
    import queue  
except ImportError:
    import Queue as queue  

CONTROL_HZ = 10.0    
LOG_HZ     = 10.0    
CONTROL_DT = 1.0 / CONTROL_HZ
LOG_DT     = 1.0 / LOG_HZ

TARGET_MAP   = "Town04_Opt"
NO_RENDERING = False
FOLLOW_IN_UE = True
SPEC_BACK    = 6.5
SPEC_UP      = 2.5

CLIENT_HOST   = "localhost"
CLIENT_PORT   = 2000
CLIENT_TIMEOUT = 30.0



class RoadOption(Enum):
    VOID=-1; LEFT=1; RIGHT=2; STRAIGHT=3; LANEFOLLOW=4; CHANGELANELEFT=5; CHANGELANERIGHT=6

def _active_map(world):
    try:
        return world.get_map().name
    except Exception:
        return "unknown"

class CarEnv:
    def __init__(self):
        self.client = carla.Client(CLIENT_HOST, CLIENT_PORT)
        self.client.set_timeout(CLIENT_TIMEOUT)

        # Wait for world
        for _ in range(120):
            try:
                self.world = self.client.get_world()
                _ = self.world.get_map()
                break
            except RuntimeError:
                time.sleep(0.5)
        else:
            raise RuntimeError("CARLA server not available")

        try: # fix reload issue, do not use the original reload in the git repo
            short = _active_map(self.world).split('/')[-1]
            if TARGET_MAP and TARGET_MAP not in short:
                print(f"[MAP] Loading map {TARGET_MAP} ...")
                self.world = self.client.load_world(TARGET_MAP)
                time.sleep(0.5)
        except Exception as e:
            print(f"[MAP] load_world warning: {e!r}")
        print(f"[MAP] Active map: {_active_map(self.world)}")

        # switch to SYNC 30 Hz
        self._original_settings = self.world.get_settings()
        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / CONTROL_HZ,
            no_rendering_mode=NO_RENDERING
        )
        self.world.apply_settings(settings)


        try:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(True)
        except Exception:
            pass

        self.bp_lib = self.world.get_blueprint_library()
        cand = self.bp_lib.filter("vehicle.tesla.model3")
        self.vehicle_bp = cand[0] if cand else self.bp_lib.filter("vehicle.*")[0]

        self.vehicle = None
        self.colsensor = None
        self.actor_list = []
        self.spectator = self.world.get_spectator()

        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

    def destroy(self):
        try:
            try:
                tm = self.client.get_trafficmanager()
                tm.set_synchronous_mode(False)
            except Exception:
                pass
            self.world.apply_settings(self._original_settings)
        except Exception:
            pass
        for a in self.actor_list[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        self.actor_list.clear()
        self.vehicle = None
        self.colsensor = None

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        spawn = random.choice(self.world.get_map().get_spawn_points())
        if self.vehicle is None:
            self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn)
            self.actor_list.append(self.vehicle)
        else:
            self.vehicle.set_transform(spawn)

        colsensor_bp = self.bp_lib.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor_bp,
            carla.Transform(carla.Location(x=2.5, z=0.7)),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda e: self.collision_hist.append(e))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self._move_spectator()
        return self.get_state()

    def _move_spectator(self):
        if NO_RENDERING or not FOLLOW_IN_UE or self.vehicle is None or self.spectator is None:
            return
        tf = self.vehicle.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        back = carla.Location(-SPEC_BACK * math.cos(yaw_rad),
                              -SPEC_BACK * math.sin(yaw_rad),
                              SPEC_UP)
        sp_tf = carla.Transform(tf.location + back,
                                carla.Rotation(pitch=-8.0, yaw=tf.rotation.yaw))
        try:
            self.spectator.set_transform(sp_tf)
        except Exception:
            pass

    def get_state(self):
        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        a = self.vehicle.get_acceleration()
        yaw = t.rotation.yaw
        ang = self.vehicle.get_angular_velocity()
        return [t.location.x, t.location.y, v.x, v.y, a.x, a.y, yaw, ang.z]

    def apply_action(self, action):
        steer, thr, brk = action
        self.steer    = float(max(-1.0, min( 1.0, steer)))
        self.throttle = float(max( 0.0, min( 1.0, thr  )))
        self.brake    = float(max( 0.0, min( 1.0, brk  )))
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=self.throttle, steer=self.steer, brake=self.brake))
        self._move_spectator()

    def tick(self):
        """Advance the world exactly one fixed 30 Hz step and return dt."""
        self.world.tick()
        return CONTROL_DT

    # unused in SYNC mode, for compat, we stop using ASYNC in the end
    def pace(self, seconds):
        pass
