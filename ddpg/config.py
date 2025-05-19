# config.py
import torch
import numpy as np

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
# NOTE: Make sure 'PandaSinusoidTrack-v0' is registered via custom_panda_env.py
ENV_NAME = 'PandaSinusoidTrack-v0'
# Max steps defined by environment registration, check custom_panda_env.py if needed

# --- Trajectory Parameters (used by custom_panda_env.py) ---
# These are defined here for reference but currently hardcoded in the env file.
# You could modify the env to read these from config if desired.
TRAJ_CENTER = np.array([0.3, 0.0, 0.2])
TRAJ_RADIUS = 0.15
TRAJ_FREQ = 0.1 # Hz
TRAJ_Z_AMP = 0.05
TRAJ_DURATION = 10.0 # seconds
CONTROL_FREQ = 20 # Hz
MAX_EPISODE_STEPS = int(TRAJ_DURATION * CONTROL_FREQ) # ~200 steps

# --- DDPG Agent ---
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99 # Dense reward, higher gamma often okay
TAU = 0.005
NOISE_STDDEV = 0.1 # Exploration noise std deviation
NOISE_CLIP = 0.5 # Clip noise range
POLICY_DELAY = 2 # TD3-like delayed policy updates
OPTIMIZER_WEIGHT_DECAY = 0.0 # L2 regularization

# --- Replay Buffer ---
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

# --- Prioritized Replay (PER) ---
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = int(1e6) # Annealing frames for beta
PER_EPSILON = 1e-6

# --- HER --- <<< REMOVED / NOT USED FOR TRAJECTORY TRACKING >>>

# --- Training ---
N_GAMES = 3000 # Total number of training episodes (adjust as needed)
WARMUP_BATCHES = 25 # Number of steps * BATCH_SIZE with random actions before training
EVAL_FREQ = 50 # Evaluate agent every N episodes
CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = "ddpg_traj_track" # Model filename prefix

# --- Evaluation ---
EVAL_EPISODES = 10
RENDER_DELAY = 50 # Delay between steps in ms for 'human' render mode (adjust for CONTROL_FREQ)
# RENDER_FILENAME = f"{MODEL_NAME}_evaluation.gif" # Default GIF filename if using rgb_array