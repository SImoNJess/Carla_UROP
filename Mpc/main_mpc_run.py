import math
import numpy as np


import carla  
import car_env as CE

CE.SHOW_CAM = False
CE.VIDEO_RECORD = False
CE.NO_RENDERING = False
CE.RES_X, CE.RES_Y = 640, 360
# try to lower down the rendering...

CE.DT_ = 0.05
CE.N_DT = 2

from ilqr_numpy_blackbox import BlackBoxDynamics, CostFunction, ilqr_optimize
from car_env import CarEnv, draw_planned_trj  

#MPC parameters
DT = 0.1           # match training 10 Hz
HORIZON = 50       # steps in each iLQR solve
APPLY_STEPS = 3    # how many optimized steps to apply before replanning
SPEED_REF = 8.0    # target speed [m/s]
MAX_ITERS = 500    # maximum MPC loop iterations

def follow_with_spectator(world, vehicle, dist=10.0, height=4.0, pitch=-12.0): #spectator
    spectator = world.get_spectator()
    tf = vehicle.get_transform()
    yaw_rad = math.radians(tf.rotation.yaw)
    back = carla.Location(x=-dist*math.cos(yaw_rad),
                          y=-dist*math.sin(yaw_rad),
                          z=height)
    look = carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw)
    spectator.set_transform(carla.Transform(tf.location + back, look))

def main():
    dyn = BlackBoxDynamics(dt=DT, lr_mean=1.5389)

    env = CarEnv()
    state, waypoints = env.reset() 
    follow_with_spectator(env.world, env.vehicle)

    for k in range(MAX_ITERS):
        if waypoints is None or len(waypoints) == 0:
            print("No waypoints left, stopping MPC loop.")
            break

        u_init = np.zeros((HORIZON-1, 3), dtype=np.float32)
        u_init[:, 1] = 0.2  

        #cost function
        cost = CostFunction(waypoints, dp=1.0, speed_ref=SPEED_REF)

        #iLQR
        x_trj, u_trj = ilqr_optimize(
            dyn, cost, state.astype(np.float32),
            u_init, max_iter=40
        )

        #visualize 
        draw_planned_trj(env.world, x_trj, env.location_[2], color=(0, 223, 222))

        for j in range(APPLY_STEPS):
            steer, thr, brk = u_trj[j]
            steer = float(np.clip(steer, -1, 1))
            thr   = float(np.clip(thr,   0, 1))
            brk   = float(np.clip(brk,   0, 1))

            state, waypoints, done, _ = env.step(np.array([steer, thr, brk]))
            #spectator
            follow_with_spectator(env.world, env.vehicle)

            print(f"[{k:03d}.{j}] v={state[2]:.2f} m/s, "
                  f"steer={steer:.2f}, thr={thr:.2f}, brk={brk:.2f}")
            if done:
                print("Episode ended (collision or timeout).")
                return

if __name__ == "__main__":
    main()
