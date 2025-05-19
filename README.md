# Robot Trajectory Tracking Projects

This repository contains three sub-projects developed as part of the **CSCI5527: Deep Learning** course at the University of Minnesota. Each project focuses on different approaches to robot trajectory tracking using deep learning and control theory.

## Subdirectories

### [rdpg](./rdpg)
Implements the Recurrent Deterministic Policy Gradient (RDPG) algorithm to train a Franka Emika Panda robot arm to follow a 3D sinusoidal trajectory. Features include:
- Recurrent actor and critic networks with LSTM
- Target networks and soft updates
- Prioritized Experience Replay (PER) storing LSTM hidden states
- Custom `panda-gym`/PyBullet environment for dense reward trajectory tracking
- Training progress visualization and evaluation metrics
- Checkpoint management for model saving/loading
- GIF demonstrations of trajectory tracking performance

### [ddpg](./ddpg)
Implements the Deep Deterministic Policy Gradient (DDPG) algorithm for the same Panda trajectory tracking task. Key aspects:
- Feedforward actor and critic networks with `tanh`-scaled continuous actions
- Gaussian exploration noise and policy delay
- Prioritized Experience Replay (PER) with Hindsight Experience Replay (HER)
- Shared custom gym environment setup with `panda-gym` and PyBullet
- Comprehensive training and evaluation scripts
- Performance visualization tools
- Model checkpointing system

### [kuka_mpc_project](./kuka_mpc_project)
Implements and compares Model Predictive Control (MPC) strategies on a simulated Kuka robot arm:
- **Linear MPC (LMPC):** Uses a linearized state-space model and QP solver
- **Nonlinear MPC (NMPC) with GRU:** Employs a learned GRU-based dynamics model within a nonlinear optimization framework
- Jupyter notebooks for interactive development and analysis
- Multiple trajectory datasets for testing and validation
- Pre-trained linear model parameters
- Comprehensive documentation of MPC implementation details

## Project Structure
Each subdirectory contains:
- Implementation code
- Configuration files
- Training and evaluation scripts
- Documentation (README.md)
- Results and visualizations
- Model checkpoints (where applicable)

## Course
This work was completed as part of the requirements for **CSCI5527: Deep Learning** at the University of Minnesota.

## Contributors
Developed by Soumith Batta, Aditya Patil, and Rishika Agarwala.