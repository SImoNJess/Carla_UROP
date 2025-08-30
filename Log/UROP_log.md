# Project Log

---

## First Week

### 1. Install CARLA
We tried to install the full version (CARLA with the Unreal Engine model, about 30GB + 120GB). However, we failed to do this as it has a high requirement of hardware dependencies, which makes it unable to run on a laptop.  
So in the end, we used a packaged version.  
The difference mainly lies in the flexibility of map editing: the packaged version is not able to create maps, while we can only use the given maps. The good thing is that the given maps are already satisfactory to our current mission.

### 2. Car Simulation
We managed to use CARLA to spawn cars at specific locations, enable route waypoint planning (by Global Route Planner), and use PID control (by Vehicle PID Control) to generate a path for a given car model.  
This allowed us to make it run through desired routes with visualized route planning and flexible spectator and sensors.

### 3. Meeting
We discussed the implementation of CARLA. For the car simulation itself, CARLA doesn't hold much importance in some functions (like steering, collision detection). Although they can be achieved by third-party libraries and dependencies, this function is not basically included in CARLA.

#### a. Next Week Goal
We are trying to implement an auto-driving overtake. So we are planning to do a simple overtake for a 2-car model next week.

---

## Second Week

### 1. Car Simulation
A simple overtake by handmade control was achieved. However, the automatic overtake still has some issues.  

I aimed for the automatic overtake directly at the initial start. The issue mainly lies in 2 portions:  
1. The FSM logic to execute overtaking depends on a scalar distance between 2 cars, which may cause confusion for a double left / second lane change, and unnecessary lane change at one side since it loses direction at the front or back logic.  
2. The Global Route Planner creates a thorough waypoint during the control, so the overtaking action makes me generate a list of waypoints outside the GRP. This slight change confuses the route planner and makes the car do a 180-degree return sometimes, trying to get back to the original waypoint before the lane change when the overtake finishes.

#### a. Next Week Goal
Fix the GRP issue, make sure the waypoint confusion doesn't happen, and use an automatic overtake to override the current handmade overtake logic.

---

## Third Week

### 1. Car Simulation
This week we managed to do an automatic overtake with no GRP and waypoint confusion.  

We tried to use Basic Agent, Behavioral Agent, and Traffic Manager to support the car model instead of the GRP. However, all failed with some issues (these 3 controls have a list of written logic for the car models inside CARLA).  

- The agents are too conservative (more AEB brake rather than overtaking in our desired condition).  
- The Traffic Manager seems to lose physical effect (similar AEB brake tendency and worse decision).  

I changed the code to make the agent bolder and the Traffic Manager more willing to overtake, however it does not satisfy our condition.  

The final solution was to get rid of these all, without GRP, and instead use **dynamic waypoints generation** at the front car. We also used vector dot products to measure relative distance between the 2 car models, which solved the multiple one-way lane change issues.

We also implemented the MATLAB data into the CARLA simulator.  
Data is in time scale with …

#### a. Next Week Goal
Improve the overtake logic to handle more severe situations (like when during overtaking, the cruiser accelerates — we may need to do a lane change back and merge later). Also, create a more impressive overtake in terms of distance (shorten the current conservative safety gap).

---

## Fourth Week

### 1. Car Simulation
Last week the FSM overtake auto model was constructed in CARLA, and we aimed to investigate further.  

Based on our original goal, I needed to do data visualization. So now what we did was to implement the MATLAB data into CARLA and enable data visualization inside CARLA.  

We had the relative position data, the angle (steering), and the speed for two cars. In CARLA, we were able to use these to generate a nice animation:  
- For each tick we teleport to the corresponding position of the car.  
- During the tick interval, we set the car speed to manipulate vehicle movements.  
Both actions need synchronization to avoid unexpected teleportation.  

Also, we needed some math scaling to convert the data into an ideal form, as the parameters may vary between two desired environments in CARLA and MATLAB simulations.

#### a. Next Week Goal
Dig deeper into the MPC model of auto drive. Learn the fundamental skills and try to understand how the model works with CARLA. 

---

## Fifth Week

### 1. Machine Learning and Car Animation Optimization
This week we went through the basic concept of MPC and the data visualization given a completed range of MATLAB data.  

For MPC, currently I just grabbed it as a way to predict the following actions. It will fetch the previous actions and predict the next 10 (or more) actions in discrete time to try matching the most logical or safest path. It will iterate each step and always take the next step, even though the predictions are for the following sequences.  

For the car animation, I managed to solve the unusual flipping issue: this was due to sudden teleportation, which caused a collision with the road that flipped the car at unexpected angles. The issue was fixed by a teleport command that fixed the car on the road with dynamic Z-axis teleport.  

Also, the mode was set back to asynchronous since the synchronous mode had a low refreshing rate, which made the whole animation movement (actually data in discrete time) laggy.

#### a. Next (after 1-week break) Week Goal
Learn more about neural networks, model construction, and the training code. Also remember that the GitHub repo is 5 years old — get ready for some version-date issues.

---

## Sixth Week

### 1. Model Construction and Neural Network Training Attempt
This week we tried to understand [Carla_iLQR_MPC](https://github.com/YukunXia/Carla_iLQR_MPC): how it works and why to choose this one.  

- The bicycle model provides a simple idea for the CARLA car simulation — it has a kinematic structure for a basic position and heading update.  
- But since the automatic car is not the same thing as a bicycle, we use a small neural network to grasp/observe its change rates of longitudinal speed and sideslip angle.  

We also tried to do training to collect the desired data for the car using the CARLA simulator. However, the fetched data returned bad results due to poor performance of my GPU laptop. QAQ  

### 2. Neural Network
**Input:**  
- Longitudinal speed (v)  
- Sideslip angle β (encoded as sinβ, cosβ instead of raw β)  
- Steering input (δ)  
- Throttle  
- Brake  

**Output:**  
- Predicted change in speed v  
- Predicted change in sideslip angle β  

### 3. Training Idea
- Take current state.  
- Predict how control sequences evolve via NN + bicycle model.  
- Compute cost (track error + control effort).  
- Optimize to find the best sequence (MPC).  
- Apply the first control to the CARLA car to see if it goes smoothly.  

The training outcome was not ideal. The computer lagged insanely when starting the pygame window, since we needed to open the client (pygame window) while keeping the server (CARLA simulator exe itself) at the same time — unacceptable on a 6GB RAM GPU.  

I tried stopping the server render, but still it didn’t work.  

The working idea for the trained data (WASD control) is the same as the manual control inside `manual_control.py`. I will try to make the pygame executable.

#### a. Next Week Goal
Solve the client data collection by some means — maybe reinstall the GPU launcher or clean everything consuming GPU before starting the CARLA simulator. Once this step is done, I can proceed with training the dynamic model (bicycle geometry + simple neural network).

---

## Seventh Week

### 1. Data Collection, iLQR and MPC Attempt (if Training Goes Well)
This week our major aim was to collect enough data for training, try running it through the iLQR loop, and make the MPC simulation via CARLA.  

We managed to get some data, however the data was quite wild due to a low refreshing rate of the CARLA simulator, which made it hard to do overall control since the FPS was too low and the brake/steer was too sensitive (unable to manipulate precisely with keyboard input).  

For GPU burnout, I managed to use the server to operate and show the vehicle spectator POV by using manual control directly sent to the server, skipping the pygame window via client. This was obviously not a good idea, since the server had severe latency on the control panel and we could only use the VSCode terminal to control. However, running an extra client independent from the server was too much for the laptop.  

We did the training via Colab for ease, using PyTorch to run the training. The training result did converge, however the output was not good — a lot of unexpected steering angles (e.g., a 180-degree turn followed by another 180-degree in the opposite direction). There must be something wrong with the data manipulation.  

Still, I tried running the trained data (through iLQR loop with MPC in CARLA to do some simple operations). The issue stayed the same, with the crashing pygame window. Python had some version mismatch with NumPy support — hopefully there’s a way to skip that.

### 2. iLQR
It takes the dynamic model (NN + bicycle), the cost function, and the current state to calculate an optimized result for the state and control.  

iLQR takes your nonlinear problem, locally pretends “the system is linear, the cost is quadratic” (→ LQR), solves that LQR optimally (backward/forward pass), applies those corrections, re-simulates with the true nonlinear dynamics, and iterates. Just like LQR on rails, gradually bending the trajectory toward optimal.

#### a. Next Week Goal
- Fix the trained data with more realistic driving operations, enhanced by smoothing the data collection window and control method.  
- Redo the data collection and Colab training.  
- Grasp a supportive NumPy version with the current Python version.  
- Do the MPC demo through CARLA with relatively fluent manipulation.

---

## Eighth Week

### 1. Data Modification and MPC Demo
This week our major aim was to improve the model training with better data collection and finish the MPC demo to observe its performance in the CARLA server.  

We found that some of the original code inside the GitHub repo was a bit sluggish for training, so we improved them (e.g., no reload, setting ASYN to SYN for stability). The changes improved functionality significantly.  

In training, we found two modifications that were inspiring:  
1. We filtered out long parking data (time when the car was accelerating with very low speed but massive steering and brake values).  
2. We forced the initial NN to start with minor tweaks, not fierce beginnings — since the sideslip angle and the velocity change are all gradual changes through time.

---
