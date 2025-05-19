# Kuka Robot Trajectory Tracking with Model Predictive Control (MPC)

## 1. Overview

This project implements and compares Model Predictive Control (MPC) strategies for trajectory tracking tasks on a Kuka robot. It features two main MPC approaches:
1.  **Linear MPC (LMPC):** Utilizes a linearized model of the robot dynamics.
2.  **Nonlinear MPC (NMPC) with a GRU-based model:** Employs a Gated Recurrent Unit (GRU) network to learn and predict the robot's nonlinear dynamics, which is then used within the NMPC framework.

The goal is to achieve accurate tracking of predefined 3D trajectories (provided as `.csv` files) by computing optimal control inputs for the Kuka robot.

## 2. Features

* **Kuka Robot Simulation:** Environment for simulating the Kuka robot (details about the simulator, e.g., PyBullet, CoppeliaSim, or a custom simulator, should be added here).
* **Trajectory Loading:** Capable of loading reference trajectories from `.csv` files.
* **Linear MPC (LMPC):**
    * Controller based on a linear state-space model of the robot.
    * Uses quadratic programming (QP) for optimization.
    * Linear model matrices (`linear_model_A.npy`, `linear_model_b.npy`) are provided.
* **Nonlinear MPC (NMPC) with GRU:**
    * Controller leveraging a learned GRU model to represent robot dynamics.
    * Handles nonlinearities more effectively than LMPC.
    * Requires a trained GRU model (details on training and model file should be added).
    * Typically involves nonlinear optimization solvers (e.g., IPOPT via CasADi).
* **Data-Driven Dynamics Modeling:** GRU network for learning robot dynamics from data.
* **Performance Evaluation:** Tools and scripts for visualizing and comparing the tracking performance of LMPC and NMPC-GRU.

## 3. System Components

* **Kuka Robot Model:** The project simulates a Kuka robot (specify model if applicable, e.g., Kuka LBR iiwa). The dynamics are either linearized for LMPC or learned by a GRU for NMPC.
* **Linear Dynamics Model:**
    * `linear_model_A.npy`: State matrix (A) for the discrete-time linear system $x_{k+1} = Ax_k + Bu_k$.
    * `linear_model_b.npy`: Input matrix (B) for the linear system.
* **GRU-based Dynamics Model:**
    * A GRU neural network trained to predict the next state of the robot given the current state and action: $x_{k+1} = f_{GRU}(x_k, u_k)$.
    * (Specify the GRU model filename, e.g., `gru_dynamics_model.pth` or `gru_model.h5`).
* **Reference Trajectories:**
    * Provided as `.csv` files (e.g., `kukatraj1.csv` to `kukatraj10.csv`).
    * Each file likely contains time-stamped waypoints for the robot's end-effector (e.g., columns for time, x, y, z position, and possibly orientation as quaternions qx, qy, qz, qw).

## 4. MPC Controllers

### 4.1. Linear MPC (LMPC)
The LMPC controller uses the predefined linear model (`linear_model_A.npy`, `linear_model_b.npy`) to predict future states. At each time step, it solves a Quadratic Program (QP) to find a sequence of control inputs that minimizes a cost function, typically penalizing tracking errors and control effort, subject to state and input constraints.

* **Script:** (e.g., `lmpc_controller.py` or `run_lmpc.py`)
* **Key Parameters:** Prediction horizon, control horizon, state weights (Q), input weights (R).

### 4.2. Nonlinear MPC (NMPC) with GRU
The NMPC controller uses the trained GRU network as its predictive model. This allows it to capture the nonlinear dynamics of the Kuka robot more accurately. At each step, a nonlinear optimization problem is solved to find the optimal control sequence.

* **GRU Model Script:** (e.g., `gru_model.py` for definition, `train_gru.py` for training)
* **NMPC Script:** (e.g., `nmpc_gru_controller.py` or `run_nmpc_gru.py`)
* **Optimization:** Typically uses tools like CasADi with solvers like IPOPT.
* **Key Parameters:** Prediction horizon, control horizon, state weights, input weights, GRU model path.

## 5. Suggested Directory Structure

This is a suggested directory structure. Please adapt it based on your actual project organization.
\`\`\`
kuka_mpc_project/
├── controllers/
│   ├── lmpc_controller.py
│   └── nmpc_gru_controller.py
├── models/
│   ├── linear_model_A.npy
│   ├── linear_model_b.npy
│   ├── gru_model.py             # GRU network definition
│   └── trained_gru_model.pth    # Example for saved GRU weights
├── trajectories/
│   ├── kukatraj1.csv
│   ├── kukatraj2.csv
│   └── ... (all 10 .csv files)
├── simulations/
│   ├── run_lmpc_simulation.py
│   └── run_nmpc_gru_simulation.py
├── utils/
│   ├── data_loader.py
│   ├── plotting.py
│   └── robot_kinematics.py      # If needed
├── training/
│   └── train_gru_model.py       # Script to train the GRU dynamics model
├── config/
│   └── mpc_params.yaml          # Or a config.py file
└── README.md
\`\`\`

## 6. Prerequisites

* Python 3.7+
* NumPy
* SciPy (for optimization, LQR components in LMPC)
* Pandas (for loading CSV trajectories)
* Matplotlib (for plotting results)
* PyTorch or TensorFlow/Keras (for the GRU model)
* **For NMPC:**
    * CasADi (highly recommended for NMPC formulation)
    * A suitable NLP solver compatible with CasADi (e.g., IPOPT, SNOPT)
* **For Simulation:**
    * (Specify simulator libraries, e.g., `pybullet`, `coppeliasim_api`)
* (Any other specific libraries used)

## 7. Setup

1.  **Clone the repository:**
    \`\`\`bash
    git clone <your-repository-url>
    cd kuka_mpc_project
    \`\`\`
2.  **Create a virtual environment (recommended):**
    * Using Conda:
        \`\`\`bash
        conda create -n kuka_mpc python=3.8
        conda activate kuka_mpc
        \`\`\`
    * Using venv:
        \`\`\`bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        \`\`\`
3.  **Install dependencies:**
    * If you have a `requirements.txt` file:
        \`\`\`bash
        pip install -r requirements.txt
        \`\`\`
    * Otherwise, install manually:
        \`\`\`bash
        pip install numpy scipy pandas matplotlib torch # or tensorflow
        pip install casadi
        (Install other specific packages like PyBullet)
        \`\`\`
    * **Note on IPOPT:** If using CasADi with IPOPT, IPOPT might need to be installed separately or via Conda. Refer to CasADi and IPOPT installation guides.

## 8. Configuration

Key parameters for the MPC controllers, simulation, and model paths should be configurable. This might be done in:
* A central `config.py` or `config.yaml` file.
* Directly within the simulation or controller scripts.

Important parameters include:
* MPC prediction and control horizons ($N_p, N_c$).
* Cost function weights (Q for states, R for inputs).
* State and control input constraints.
* Paths to the linear model files (`.npy`).
* Path to the trained GRU model.
* Selected trajectory file.
* Simulation parameters (e.g., time step, duration).

## 9. How to Run

*(Please provide specific commands and script names based on your project)*

### 9.1. Training the GRU Dynamics Model (if applicable)
If the GRU model needs to be trained or re-trained:
\`\`\`bash
python training/train_gru_model.py --data <path_to_training_data> --epochs <num_epochs>
\`\`\`

### 9.2. Running LMPC Simulation
To run a simulation with the Linear MPC:
\`\`\`bash
python simulations/run_lmpc_simulation.py --trajectory trajectories/kukatraj1.csv --config config/mpc_params.yaml
\`\`\`

### 9.3. Running NMPC-GRU Simulation
To run a simulation with the Nonlinear MPC using the GRU model:
\`\`\`bash
python simulations/run_nmpc_gru_simulation.py --trajectory trajectories/kukatraj1.csv --gru_model models/trained_gru_model.pth --config config/mpc_params.yaml
\`\`\`

## 10. Trajectory Data (`kukatraj*.csv`)

The `kukatraj*.csv` files contain the reference trajectories for the Kuka robot. It's assumed they have the following format (please verify and update):
* **Columns:** `time, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w`
    * `time`: Time in seconds.
    * `pos_x, pos_y, pos_z`: Target end-effector position in meters.
    * `quat_x, quat_y, quat_z, quat_w`: Target end-effector orientation as a quaternion.
    *(Alternatively, if only position tracking is performed, orientation columns might be absent or ignored).*

## 11. Expected Results

Upon running the simulations, you should expect:
* Console output logging the simulation progress and performance metrics (e.g., Mean Squared Error for tracking).
* Plots generated (e.g., in a `results/` directory) showing:
    * Comparison of the robot's actual end-effector path versus the reference trajectory in 3D or per-axis.
    * Tracking error over time.
    * Control inputs applied by the MPC.
* If comparing LMPC and NMPC-GRU, plots or metrics highlighting their relative performance.

## 12. Troubleshooting / Notes

* **Solver Issues (NMPC):** Ensure that CasADi and the chosen NLP solver (e.g., IPOPT) are correctly installed and accessible in your Python environment. Solver convergence can be sensitive to model accuracy, cost function formulation, and initial guesses.
* **GRU Model Performance:** The NMPC-GRU's effectiveness heavily depends on the accuracy of the trained GRU model. Ensure the GRU model is well-trained on representative data.
* **Computational Cost:** NMPC is generally more computationally intensive than LMPC, especially with longer prediction horizons.
* (Add any other specific notes or common issues encountered during development).
