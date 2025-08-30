# ==== car_env.py (headless/spectator-ready) ====
# Works without a local pygame window. The UE server's spectator camera is used to observe.
# You can flip SHOW_CAM=True later if you want to bring back a pygame client + RGB camera.

from collections import deque
import numpy as onp
import random
from enum import Enum
import math
import itertools

from tqdm.auto import tqdm

try:
    import queue
except ImportError:
    import Queue as queue

import carla

# Try pygame, but only required if SHOW_CAM=True
try:
    import pygame
    pygame.init()
except Exception:
    pygame = None  # ok when SHOW_CAM=False

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

DT_ = 0.05          # [s] world fixed delta (20 Hz) → gentler load than 0.01
N_DT = 2            # ticks per env.step → 0.1 s per env.step (aligns with MPC DT=0.1)

NO_RENDERING = False          # keep server rendering ON so you can watch via spectator
WAYPOINT_BUFFER_LEN = 100
WAYPOINT_INTERVAL = 1         # [m]
WAYPOINT_BUFFER_MID_INDEX = int(WAYPOINT_BUFFER_LEN/2)

FUTURE_WAYPOINTS_AS_STATE = 50

SHOW_CAM = False              # <<< headless by default (no pygame window, no RGB camera)
START_TIME = 3
DEBUG = True
MPC_INTERVAL = 1

VIDEO_RECORD = False          # PNG saving off by default (you can re-enable later)
RES_X = 640                   # used only if SHOW_CAM=True
RES_Y = 360

# ==============================================================================
# -- Utilities -----------------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = onp.frombuffer(image.raw_data, dtype=onp.dtype("uint8"))
    array = onp.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_waypoints(world, waypoints, z=0.5, color=(255,0,0)):
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z)
        world.debug.draw_point(begin, size=0.05, color=color, life_time=0.1)

def draw_planned_trj(world, x_trj, car_z, color=(255,0,0)):
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    length = x_trj.shape[0]
    xx = x_trj[:,0]; yy = x_trj[:,1]
    for i in range(1, length):
        begin = carla.Location(float(xx[i-1]), float(yy[i-1]), float(car_z+1))
        end   = carla.Location(float(xx[i]),   float(yy[i]),   float(car_z+1))
        world.debug.draw_line(begin=begin, end=end, thickness=0.1, color=color, life_time=0.1*MPC_INTERVAL)

class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

def get_font():
    if pygame is None:
        return None
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else (fonts[0] if fonts else None)
    if font is None:
        return None
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def _retrieve_options(list_waypoints, current_waypoint):
    options = []
    for next_waypoint in list_waypoints:
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)
    return options

def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    n = next_waypoint.transform.rotation.yaw % 360.0
    c = current_waypoint.transform.rotation.yaw % 360.0
    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

# ==============================================================================
# -- Environment ---------------------------------------------------------------
# ==============================================================================

class CarEnv:

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # IMPORTANT: attach to the existing world (don't force reload here)
        MAP_NAME = "Town02_Opt"     # <- change here
        self.world = self.client.load_world(MAP_NAME)
        # self.world = self.client.reload_world()  # avoid heavy blocking reload inside __init__

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle = None
        self.actor_list = []

        # Sync world with fixed step
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING,
            synchronous_mode=True,
            fixed_delta_seconds=DT_
        ))

        # Waypoint buffer
        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)

        # Pygame/display only if SHOW_CAM
        if SHOW_CAM:
            assert pygame is not None, "pygame is required when SHOW_CAM=True"
            self.display = pygame.display.set_mode(
                (RES_X, RES_Y), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.font = get_font()
        else:
            self.display = None
            self.font = None

        # Always keep a clock to rate-limit when SHOW_CAM=True
        self.clock = _SimpleClock()

        self.file_num = 0

    def reset(self):
        tqdm.write("call reset")

        self.collision_hist = []
        self.actor_list = []

        # fresh spawn each episode
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())

        new_car = (self.vehicle is None)
        if new_car:
            self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
            self.actor_list.append(self.vehicle)
        else:
            self.vehicle.set_transform(self.spawn_point)

        # Collision sensor (always on)
        if new_car:
            transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))

        # RGB camera + queue only if SHOW_CAM
        if SHOW_CAM and new_car:
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '{}'.format(RES_X))
            blueprint.set_attribute('image_size_y', '{}'.format(RES_Y))
            blueprint.set_attribute('sensor_tick', str(DT_))  # capture each tick
            self.camera = self.world.spawn_actor(
                blueprint,
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera)
            self.image_queue = queue.Queue(maxsize=1)  # drop old frames
            def _on_image(img):
                if not self.image_queue.full():
                    self.image_queue.put(img)
            self.camera.listen(_on_image)
        else:
            self.camera = None
            self.image_queue = None

        # Waypoints
        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)
        self.update_waypoint_buffer(given_loc=[True, self.spawn_point.location])

        self.time = 0.0
        return self.get_state(), self.get_waypoint()

    def collision_data(self, event):
        self.collision_hist.append(event)

    def update_waypoint_buffer(self, given_loc = [False, None]):
        if given_loc[0]:
            car_loc = given_loc[1]
        else:
            car_loc = self.vehicle.get_location()

        self.min_distance = onp.inf
        if (len(self.waypoint_buffer) == 0):
            self.waypoint_buffer.append(self.map.get_waypoint(car_loc))

        for i in range(len(self.waypoint_buffer)):
            curr_distance = self.waypoint_buffer[i].transform.location.distance(car_loc)
            if curr_distance < self.min_distance:
                self.min_distance = curr_distance
                min_distance_index = i

        num_waypoints_to_be_added = max(0, min_distance_index - WAYPOINT_BUFFER_MID_INDEX)
        num_waypoints_to_be_added = max(num_waypoints_to_be_added, WAYPOINT_BUFFER_LEN - len(self.waypoint_buffer))

        for _ in range(num_waypoints_to_be_added):
            frontier = self.waypoint_buffer[-1]
            next_waypoints = list(frontier.next(WAYPOINT_INTERVAL))
            if len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = _retrieve_options(next_waypoints, frontier)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]
            self.waypoint_buffer.append(next_waypoint)

        self.min_distance_index = WAYPOINT_BUFFER_MID_INDEX if min_distance_index > WAYPOINT_BUFFER_MID_INDEX else min_distance_index

    def get_state(self):
        self.location = self.vehicle.get_location()
        self.location_ = onp.array([self.location.x, self.location.y, self.location.z])

        self.transform = self.vehicle.get_transform()
        phi = self.transform.rotation.yaw*onp.pi/180  # yaw (rad)

        self.velocity = self.vehicle.get_velocity()
        vx = self.velocity.x; vy = self.velocity.y

        beta_candidate = onp.arctan2(vy, vx) - phi + onp.pi*onp.array([-2,-1,0,1,2])
        local_diff = onp.abs(beta_candidate - 0)
        min_index = onp.argmin(local_diff)
        beta = beta_candidate[min_index]

        state = [ self.location.x,
                  self.location.y,
                  onp.sqrt(vx**2 + vy**2),
                  phi,
                  beta ]
        return onp.array(state)

    def get_waypoint(self):
        waypoints = []
        for i in range(self.min_distance_index, self.min_distance_index+FUTURE_WAYPOINTS_AS_STATE):
            waypoint_location = self.waypoint_buffer[i].transform.location
            waypoints.append([waypoint_location.x, waypoint_location.y])
        return onp.array(waypoints)

    def step(self, action):
        assert len(action) == 3

        if self.time >= START_TIME:
            steer_, throttle_, brake_ = action
        else:
            steer_ = 0.0
            throttle_ = 0.5
            brake_ = 0.0

        assert -1.0 <= steer_ <= 1.0 and 0.0 <= throttle_ <= 1.0 and 0.0 <= brake_ <= 1.0

        tqdm.write("steer = {0:5.2f}, throttle {1:5.2f}, brake {2:5.2f}".format(
            float(steer_), float(throttle_), float(brake_)))

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_),
                                                       steer=float(steer_),
                                                       brake=float(brake_)))

        for i in range(N_DT):
            # keep sim in sync
            self.clock.tick()
            self.world.tick()
            self.time += DT_

            # If we’re showing the local client window, pump events and draw the camera
            if SHOW_CAM and self.image_queue is not None:
                # keep OS happy
                if pygame is not None:
                    pygame.event.pump()
                try:
                    image_rgb = self.image_queue.get(timeout=1.0)  # don’t block forever
                    draw_image(self.display, image_rgb)
                    if self.font is not None:
                        vel = self.vehicle.get_velocity()
                        self.display.blit(
                            self.font.render('Velocity = {0:.2f} m/s'.format(
                                math.sqrt(vel.x**2 + vel.y**2)), True, (255, 255, 255)), (8, 10))
                        # simple control bars
                        v_offset = 25
                        bar_h_offset = 75
                        bar_width = 100
                        for key, value in {"steering":steer_, "throttle":throttle_, "brake":brake_}.items():
                            rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                            pygame.draw.rect(self.display, (255, 255, 255), rect_border, 1)
                            if key == "steering":
                                rect = pygame.Rect((bar_h_offset + (1+value) * (bar_width)/2, v_offset + 8), (6, 6))
                            else:
                                rect = pygame.Rect((bar_h_offset + value * (bar_width), v_offset + 8), (6, 6))
                            pygame.draw.rect(self.display, (255, 255, 255), rect)
                            self.display.blit(self.font.render(key, True, (255,255,255)), (8, v_offset+3))
                            v_offset += 18
                    if pygame is not None:
                        pygame.display.flip()
                    if VIDEO_RECORD and self.time >= START_TIME and pygame is not None:
                        filename = "Snaps/%05d.png" % self.file_num
                        pygame.image.save(self.display, filename)
                        self.file_num += 1
                except queue.Empty:
                    pass  # skip drawing this tick if no frame

        # Update waypoints and debug draw
        self.update_waypoint_buffer()
        if DEBUG:
            past_WP = list(itertools.islice(self.waypoint_buffer, 0, self.min_distance_index))
            future_WP = list(itertools.islice(self.waypoint_buffer, self.min_distance_index+1, WAYPOINT_BUFFER_LEN-1))
            draw_waypoints(self.world, future_WP, z=0.5, color=(255,0,0))
            draw_waypoints(self.world, past_WP,   z=0.5, color=(0,255,0))
            draw_waypoints(self.world, [self.waypoint_buffer[self.min_distance_index]], z=0.5, color=(0,0,255))

        done = (len(self.collision_hist) != 0)
        new_state = self.get_state()
        waypoints = self.get_waypoint()
        return new_state, waypoints, done, None


# Small cross-version clock that never hard-depends on pygame
class _SimpleClock:
    def __init__(self):
        self._last = None
    def tick(self):
        # Keep method for API compatibility; we don't throttle here intentionally.
        return
