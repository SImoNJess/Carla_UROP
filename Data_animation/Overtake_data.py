import carla
import time
import numpy as np
import math
import sys
import os
import scipy.io
import msvcrt

python_api = r'D:/UE4-Carla/sim/PythonAPI'
sys.path.append(python_api)
sys.path.append(python_api + '/carla')

SIM_TIME_STEP = 0.1
CLIENT_TIMEOUT = 20.0
UPSAMPLE_FACTOR = 10
# do this to insert extra variable between
SIM_TIME_STEP = SIM_TIME_STEP / UPSAMPLE_FACTOR

mat_file = 'carla_data.mat'
if os.path.isfile(mat_file):
    print(f"Loading data from {mat_file}")
    mat = scipy.io.loadmat(mat_file)
    xx = mat['xx']
    x_rel = xx[0, :]
    y_rel = xx[1, :]
    yaw_rel = xx[2, :]
    speed_overtaker = xx[3, :]
    speed_cruiser = xx[4, :]
    T = x_rel.size
else:
    print('Fallback data')
    T = 100
    x_rel = np.full(T, 5.0)
    y_rel = np.linspace(0, 3.5, T)
    yaw_rel = np.zeros(T)
    speed_overtaker = np.full(T, 10.0)
    speed_cruiser = np.full(T, 10.0)

def upsample(arr, factor):
    n = arr.shape[0]
    orig_idx = np.arange(n)
    new_idx = np.linspace(0, n - 1, (n - 1) * factor + 1)
    return np.interp(new_idx, orig_idx, arr)


x_rel_up = upsample(x_rel, UPSAMPLE_FACTOR)
y_rel_up = upsample(y_rel, UPSAMPLE_FACTOR)
yaw_rel_up = upsample(yaw_rel, UPSAMPLE_FACTOR)
speed_overtaker_up = upsample(speed_overtaker, UPSAMPLE_FACTOR)
speed_cruiser_up = upsample(speed_cruiser, UPSAMPLE_FACTOR)
T_up = x_rel_up.size


client = carla.Client('localhost', 2000)
client.set_timeout(CLIENT_TIMEOUT)
world = client.get_world()
if world.get_map().name != 'Town06_Opt': 
    world = client.load_world('Town06_Opt')
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

spawn_pts = world.get_map().get_spawn_points()
base_spawn = spawn_pts[55] #for map6, try 328 180 55 for map4, try 242
base_wp = world.get_map().get_waypoint(base_spawn.location)
origin_yaw = math.radians(base_spawn.rotation.yaw)
axis_lateral = carla.Vector3D(-math.sin(origin_yaw), math.cos(origin_yaw), 0)

vehicle_bp = world.get_blueprint_library().filter('vehicle.audi.a2')[0]
fwd_vec = base_wp.transform.rotation.get_forward_vector()
back_loc_est = base_wp.transform.location - fwd_vec * 50.0
back_wp = world.get_map().get_waypoint(back_loc_est, project_to_road=True)
cruiser = world.spawn_actor(vehicle_bp, back_wp.transform)
cruiser.set_simulate_physics(False)

init_x = x_rel_up[0]; init_y = y_rel_up[0]
o_loc0 = back_wp.transform.location + fwd_vec * init_x + axis_lateral * init_y
init_yaw = math.degrees(math.radians(back_wp.transform.rotation.yaw) + yaw_rel_up[0])

ego = world.spawn_actor(vehicle_bp, carla.Transform(
    carla.Location(o_loc0.x, o_loc0.y, back_wp.transform.location.z),
    carla.Rotation(yaw=init_yaw)
))
ego.set_simulate_physics(False)

def continuous_with_z(world_map, x_rel, y_rel, origin_loc, origin_yaw):
    dx = x_rel * math.cos(origin_yaw) - y_rel * math.sin(origin_yaw)
    dy = x_rel * math.sin(origin_yaw) + y_rel * math.cos(origin_yaw)
    loc = carla.Location(x=origin_loc.x + dx,
                         y=origin_loc.y + dy,
                         z=origin_loc.z)  # initial z
    #z project
    wp = world_map.get_waypoint(loc, project_to_road=True)
    loc.z = wp.transform.location.z
    return loc


spec = world.get_spectator()
spec.set_transform(carla.Transform(
    ego.get_transform().location + carla.Location(z=30),
    ego.get_transform().rotation
))


try:
    origin_loc = back_wp.transform.location
    i = 0
    while i < T_up:
        if msvcrt.kbhit() and msvcrt.getch().lower() == b'q': break

        # cruiser update
        if i>0:
            wp=world.get_map().get_waypoint(origin_loc,project_to_road=True)
            origin_loc+=wp.transform.get_forward_vector()*speed_cruiser_up[i]*SIM_TIME_STEP

        wp_cr = world.get_map().get_waypoint(origin_loc, project_to_road=True)
        loc_cr = wp_cr.transform.location
        rot_cr = wp_cr.transform.rotation
        cruiser.set_transform(carla.Transform(loc_cr, rot_cr))

        # overtaker update
        origin_yaw = math.radians(rot_cr.yaw)
        loc_ot = continuous_with_z(world.get_map(), x_rel_up[i], y_rel_up[i], loc_cr, origin_yaw)
        yaw_deg = math.degrees(origin_yaw + yaw_rel_up[i])
        ego.set_transform(carla.Transform(loc_ot, carla.Rotation(yaw=yaw_deg)))

        # spectator
        spec.set_transform(carla.Transform(
            ego.get_transform().location + carla.Location(z=40),
            carla.Rotation(pitch=-90, yaw=-180,roll=0)
        ))

        # # For driver POV
        # spec.set_transform(carla.Transform(
        #     ego.get_transform().location + carla.Location(x=-0.5,y=-0.3,z=1.8),
        #     ego.get_transform().rotation
        # ))

        world.wait_for_tick()
        time.sleep(SIM_TIME_STEP)
        i += 1
finally:
    ego.destroy(); cruiser.destroy()
