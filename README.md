# Carla UROP
This project is part of the Summer UROP at Imperial, focusing on CARLA investigation.

# CARLA Autonomous Driving Experiments

This repository collects a series of experiments and implementations using the [CARLA Simulator](https://carla.org/).  
It includes simple safety/decision modules, MATLAB animations, MPC training and testing pipelines, and experiment logs.  

---

## Working Environment

- **Simulator Version:** CARLA 0.9.15 (UE4) [Release](https://github.com/carla-simulator/carla/releases)
- **OS:** Windows 11  
- **GPU:** NVIDIA RTX 3050 Laptop GPU (6GB VRAM)  
- **CPU:** Intel i5-13500H (ultrabook laptop)  
- **Python:** 3.7.9   
- **Note:** The repository is adapted for limited GPU performance, and some implementations avoid the full client rendering overhead.  

---

## 1. Simple CARLA Files

### `AEB.py`  
- Implements a basic **Automatic Emergency Braking (AEB)** system.  
- Monitors the ego vehicle and triggers braking when a collision risk is detected. [need to spawn a car along the cruisor reference path]

### `Overtake_Dynamic.py`  
- Demonstrates a simple **autonomous overtaking maneuver** in CARLA.  
- Simple FSM to do an overtake following the lane.  

** [Demo Video](videos/MPC.mp4) **

---

## 2. MATLAB Data Animation Files

- These scripts load `.mat` data and visualize animations of vehicle trajectories (e.g., lane change, overtaking scenarios).  
- Mainly used for playback and visualization of planned maneuvers outside the simulator. 

### `Overtake_data.py`  
- The current data file presents an overtake.

** [Demo Video](videos/MPC.mp4) **

---

## 3. MPC Training and Testing

This part is based on [Carla_iLQR_MPC](https://github.com/YukunXia/Carla_iLQR_MPC).  
- The original repository provides iLQR-based MPC for trajectory planning and control.  
- In this repo, the code is **modified to run it possible on laptops with limited GPU performance**:  
  - Excludes CARLA client visualization. (no pygame window)
  - Runs fully on server-side simulation.  
  - Simplified setup for faster iteration. (no map reload at each call)

### Contents:  
- **Training files** 
`model_data_collection_main.py` `model_data_collection_preparation.py` (for Neural Network traning data collection and further iLQR optimization).  
- **Colab AI training files** 
`train+test.py` (for cloud-based training, in pytorch, NN).  
- **Trained MPC output files** 
`ilqr_numpy_blackbox.py` `dyn_head_inference.py` (model and ilqr computation in pytorch) 
`main_mpc_run.py` `car_env.py` (computes the ilqr and animation at the same moment) 
`mpc_plan.py` `mpc_replay.py`(wraps the ilqr first, then replay for animation to rescue GPU).  

** [Demo Video](videos/MPC.mp4) **

---

## 4. Data examples
- A basic model in 10 Hz collection.

---

## 5. Logs

- Includes weekly experiment logs.
- Records issues, fixes, and progress across different experiments.  



