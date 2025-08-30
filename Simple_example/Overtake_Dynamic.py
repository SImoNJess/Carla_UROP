import math
import carla
import time
import sys
import msvcrt
from types import SimpleNamespace

python_api = r'D:/UE4-Carla/sim/PythonAPI'
sys.path.append(python_api)
sys.path.append(python_api + '/carla')

from agents.navigation.controller import VehiclePIDController  # type:ignore

START_IDX           = 120  # dummy spawn point
OVT_START           = 242  # overtaker spawn point
CRUISE_SPEED        = 30.0   # dummy
OVERTAKE_SPEED      = 60.0   # overtaker
LANE_CHANGE_SPEED   = 51.0
DETECT_RANGE        = 20.0
SAFETY_GAP          = 5.0
SAMPLE_RES          = 2.0
LANE_CHANGE_AHEAD   = 25.0   
RETURN_AHEAD        = 30.0   
MERGE_DISTANCE      = 5.0   
DRAW_AHEAD_COUNT    = 10     
LANE_CHANGE_DISTANCES = (5.0, 10.0, 15.0, 20.0)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    if world.get_map().name != 'Town04_Opt':
        world = client.load_world('Town04_Opt')
        print('Loaded map:', world.get_map().name)
    else:
        print('Already on map:', world.get_map().name)

    m = world.get_map()
    sp = m.get_spawn_points()

    # Spawn dummy vehicle
    dummy_tf = sp[START_IDX]
    dummy_bp = world.get_blueprint_library().find('vehicle.audi.a2')
    dummy   = world.spawn_actor(dummy_bp, dummy_tf)
    print('Spawned dummy at #', START_IDX)

    # Dummy controller (lane-following)
    dummy_ctrl = VehiclePIDController(
        dummy,
        {'K_P':1.95,'K_I':0.07,'K_D':0.2,'dt':1/20.0},
        {'K_P':1.0,'K_I':0.75,'K_D':0.0,'dt':1/20.0}
    )

    # spawn overtaker
    ovt_tf    = sp[OVT_START]
    ovt_bp    = world.get_blueprint_library().find('vehicle.tesla.cybertruck')
    overtaker = world.spawn_actor(ovt_bp, ovt_tf)
    print('Spawned overtaker at #', OVT_START)

    # spawn spectator
    spectator = world.get_spectator()
    init_loc  = overtaker.get_transform().location + carla.Location(z=30)
    init_rot  = carla.Rotation(pitch=-90, yaw=180, roll=0)
    init_car = overtaker.get_transform()
    init_rot = init_car.rotation
    spectator.set_transform(carla.Transform(init_loc, init_rot))

    ovt_ctrl = VehiclePIDController(
        overtaker,
        {'K_P':1.95,'K_I':0.07,'K_D':0.2,'dt':1/20.0},
        {'K_P':1.0,'K_I':0.75,'K_D':0.0,'dt':1/20.0}
    )

    # State flags for overtaker
    overtaking      = False
    ready_to_merge  = False
    returning       = False
    lane_change_wps = []
    lc_index        = 0
    target_wp       = None

    try:
        print('Press Q to quit.')
        while True:
            if msvcrt.kbhit() and msvcrt.getch().lower() == b'q':
                print('Exit requested.')
                break

            # --- Dummy lane-following logic ---
            cw_d = m.get_waypoint(dummy.get_transform().location)
            next_wps = cw_d.next(SAMPLE_RES)
            if next_wps:
                wp_target = next_wps[0]
                ctrl_d = dummy_ctrl.run_step(CRUISE_SPEED, wp_target)
                dummy.apply_control(ctrl_d)

            # get transforms and distance for overtaking logic
            my_tf   = overtaker.get_transform()
            lead_tf = dummy.get_transform()
            dist_xy = math.hypot(
                lead_tf.location.x - my_tf.location.x,
                lead_tf.location.y - my_tf.location.y
            )

            # start left overtake
            if (not overtaking and not returning and not ready_to_merge
                    and not lane_change_wps and dist_xy < DETECT_RANGE):
                cw  = m.get_waypoint(my_tf.location)
                pts = cw.next(LANE_CHANGE_AHEAD)
                if pts:
                    start_wp = pts[0]
                    left_wps = []
                    for d in LANE_CHANGE_DISTANCES:
                        sub = start_wp.next(d)
                        if sub:
                            lw = sub[0].get_left_lane()
                            if lw and lw.road_id == start_wp.road_id:
                                left_wps.append(lw)
                    if left_wps:
                        lane_change_wps = left_wps
                        overtaking      = True
                        lc_index        = 0
                        target_wp       = lane_change_wps[0]
                        print('Starting left lane change.')

            # left lane change
            elif overtaking and lane_change_wps:
                ctrl_o = ovt_ctrl.run_step(LANE_CHANGE_SPEED, target_wp)
                overtaker.apply_control(ctrl_o)
                if overtaker.get_transform().location.distance(target_wp.transform.location) < 1.5:
                    lc_index += 1
                    if lc_index < len(lane_change_wps):
                        target_wp = lane_change_wps[lc_index]
                    else:
                        lane_change_wps.clear()

            # cruise in left lane
            elif overtaking and not lane_change_wps:
                overtaking     = False
                ready_to_merge = True
                print('Cruising in left lane.')

            # overtaker normal cruising and merge-back logic
            elif not overtaking and not returning and not lane_change_wps:
                cw  = m.get_waypoint(my_tf.location)
                nxt_list = cw.next(SAMPLE_RES)
                if nxt_list:
                    target_wp = nxt_list[0]
                    ctrl_o    = ovt_ctrl.run_step(OVERTAKE_SPEED, target_wp)
                    overtaker.apply_control(ctrl_o)

                if ready_to_merge:
                    rel_x   = my_tf.location.x - lead_tf.location.x
                    rel_y   = my_tf.location.y - lead_tf.location.y
                    fwd     = my_tf.rotation.get_forward_vector()
                    ahead_d = rel_x * fwd.x + rel_y * fwd.y
                    if ahead_d > MERGE_DISTANCE and dist_xy > SAFETY_GAP:
                        print('Merge conditions met; returning...')
                        cw2     = m.get_waypoint(my_tf.location)
                        pts2    = cw2.next(RETURN_AHEAD)
                        if pts2:
                            merge_wp = pts2[0]
                            right_wps = []
                            for d in LANE_CHANGE_DISTANCES:
                                sub = merge_wp.next(d)
                                if sub:
                                    rw = sub[0].get_right_lane()
                                    if rw and rw.road_id == merge_wp.road_id:
                                        right_wps.append(rw)
                            if right_wps:
                                lane_change_wps = right_wps
                                returning       = True
                                ready_to_merge  = False
                                lc_index        = 0
                                target_wp       = lane_change_wps[0]

            # right lane change
            elif returning and lane_change_wps:
                ctrl_o = ovt_ctrl.run_step(LANE_CHANGE_SPEED, target_wp)
                overtaker.apply_control(ctrl_o)
                if overtaker.get_transform().location.distance(target_wp.transform.location) < 1.5:
                    lc_index += 1
                    if lc_index < len(lane_change_wps):
                        target_wp = lane_change_wps[lc_index]
                    else:
                        lane_change_wps.clear()

            # after return finish, normal cruise
            elif returning and not lane_change_wps:
                returning = False
                print('Returned to right lane.')

            # visualization
            debug = world.debug
            if lane_change_wps:
                trace_wps = lane_change_wps[lc_index:lc_index+DRAW_AHEAD_COUNT]
                color     = carla.Color(0,255,0)
            else:
                trace_wps = []
                cw_vis    = m.get_waypoint(my_tf.location)
                for _ in range(DRAW_AHEAD_COUNT):
                    next_wps = cw_vis.next(SAMPLE_RES)
                    if next_wps:
                        trace_wps.append(next_wps[0])
                        cw_vis = next_wps[0]
                color = carla.Color(0,0,255)
            for wp in trace_wps:
                debug.draw_point(wp.transform.location + carla.Location(z=1),
                                 size=0.1, life_time=0.1, color=color)

            # TOP POV
            spec_tf = carla.Transform(
                my_tf.location + carla.Location(z=40),
                carla.Rotation(pitch=-90,yaw=-180,roll=0)
            )
            # # Driver POV
            # spec_tf = carla.Transform(
            #     my_tf.location + carla.Location(x=-0.5,y=-0.3,z=1.8),
            #     init_rot
            # )

            spectator.set_transform(spec_tf)

            world.wait_for_tick()

    finally:
        dummy.destroy()
        overtaker.destroy()
        print('Cleaned up vehicles.')

if __name__ == '__main__':
    main()
