# custom_panda_env.py
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
import traceback
try:
    import pybullet as p # Import pybullet for drawing
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: pybullet package not found. Debug visualizations will be disabled.")


# Attempt to import panda_gym components safely
try:
    from panda_gym.envs.core import RobotTaskEnv, Task
    from panda_gym.envs.robots.panda import Panda
    from panda_gym.pybullet import PyBullet
except ImportError as e:
    # ... (error handling) ...
    raise

# --- Define trajectory parameters ---
TRAJ_CENTER = np.array([0.3, 0.0, 0.2])
TRAJ_RADIUS = 0.15
TRAJ_FREQ = 0.1 # Hz (10 seconds for one full circle)
TRAJ_Z_AMP = 0.05
TRAJ_DURATION = 10.0 # seconds
SIM_FREQ = 240 # Hz (Pybullet simulation frequency)
CONTROL_FREQ = 20 # Hz (Agent action frequency)
MAX_EPISODE_STEPS = int(TRAJ_DURATION * CONTROL_FREQ)

# Trajectory function
def get_sinusoid_trajectory_point(t, cx=TRAJ_CENTER[0], cy=TRAJ_CENTER[1], cz=TRAJ_CENTER[2], r=TRAJ_RADIUS, f=TRAJ_FREQ, az=TRAJ_Z_AMP):
    angle = 2 * np.pi * f * t
    target_x = cx + r * np.cos(angle)
    target_y = cy + r * np.sin(angle)
    target_z = cz + az * np.sin(2 * angle)
    return np.array([target_x, target_y, target_z])

# --- Define the Task Class ---
# (SinusoidTrackTask class definition remains IDENTICAL to the previous working version)
class SinusoidTrackTask(Task):
    def __init__(self, sim, robot):
        super().__init__(sim)
        self.robot = robot
        self.sim_time = 0.0
        self.goal_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.goal = self._get_goal()

    def _get_goal(self):
        calc_time = min(self.sim_time, TRAJ_DURATION)
        return get_sinusoid_trajectory_point(calc_time)

    def get_obs(self): return self.goal.astype(np.float32)
    def get_achieved_goal(self):
        if hasattr(self.robot, 'get_ee_position'): return np.array(self.robot.get_ee_position()).astype(np.float32)
        else: print("Error: Task's robot object missing get_ee_position."); return np.zeros(3, dtype=np.float32)
    def compute_reward(self, achieved_goal, desired_goal, info): return -np.sum((np.array(achieved_goal) - np.array(desired_goal))**2)
    def is_success(self, achieved_goal, desired_goal, info=None): return False
    def reset(self): self.sim_time = 0.0; self.goal = self._get_goal(); return self.get_obs()


# --- Define the Environment Class ---
class PandaSinusoidTrackEnv(RobotTaskEnv):
    """
    Environment combining the Panda robot with the SinusoidTrackTask.
    Includes visualization of the target trajectory in human render mode.
    """
    def __init__(self, render_mode=None, control_type="ee"):
        try:
            self.sim_time = 0.0
            self.current_step = 0
            self._control_type = control_type
            self.sim_steps_per_control = SIM_FREQ // CONTROL_FREQ
            # Store render_mode passed during creation
            self.render_mode = render_mode
            print(f"DEBUG: PandaSinusoidTrackEnv Initializing with render_mode='{self.render_mode}'") # Debug Print 1

            # --- Visualization attributes ---
            self.target_marker_id = None
            self.trajectory_line_ids = []
            # -----------------------------

            # Initialize simulation, passing the render_mode
            sim = PyBullet(render_mode=self.render_mode, n_substeps=self.sim_steps_per_control)
            self.sim = sim
            robot = Panda(sim=self.sim, block_gripper=True, control_type=self._control_type)
            self.robot = robot
            task = SinusoidTrackTask(sim=self.sim, robot=self.robot)
            self.task = task

            # Initialize RobotTaskEnv base class
            super().__init__(robot=self.robot, task=self.task)

            self.control_frequency = CONTROL_FREQ

            # --- Define observation_space AFTER super().__init__() ---
            # Infer robot observation shape
            try:
                robot_obs_sample = self.robot.get_obs()
                if isinstance(robot_obs_sample, dict): robot_obs_vector = robot_obs_sample.get("observation", np.array([]))
                else: robot_obs_vector = robot_obs_sample
                robot_obs_shape = robot_obs_vector.flatten().shape
                if not robot_obs_shape or robot_obs_shape[0] == 0: raise ValueError("Inferred robot observation shape is invalid.")
            except Exception as e_get_obs: raise AttributeError("Could not determine robot observation space shape.") from e_get_obs
            robot_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=robot_obs_shape, dtype=np.float32)
            if not hasattr(self.task, 'goal_space'): raise AttributeError("Task object missing goal_space.")
            self.observation_space = gym.spaces.Dict(dict(observation=robot_observation_space, achieved_goal=self.task.goal_space, desired_goal=self.task.goal_space))

            # --- Define action space AFTER super().__init__() ---
            if not hasattr(self, 'action_space'):
                if hasattr(self.robot, 'action_space'): self.action_space = self.robot.action_space
                else: raise AttributeError("Environment has no action_space defined.")

        except Exception as e:
             # ... (Error handling) ...
             print("\n*** Error during PandaSinusoidTrackEnv __init__ ***"); traceback.print_exc(); print("*"*51+"\n")
             if hasattr(self, 'sim') and self.sim is not None:
                  try: self.sim.close()
                  except Exception: pass
             raise e

    def _get_target_position(self): # Renamed from _get_target_from_task for clarity
        """Helper to get target position based on environment's time."""
        calc_time = min(self.sim_time, TRAJ_DURATION)
        return get_sinusoid_trajectory_point(calc_time)

    def _get_obs(self):
        """Return observation dictionary."""
        # ... (Implementation remains the same) ...
        robot_obs_raw = self.robot.get_obs(); robot_obs_vector = robot_obs_raw.get("observation", robot_obs_raw) if isinstance(robot_obs_raw, dict) else robot_obs_raw
        achieved_goal = self.task.get_achieved_goal(); desired_goal = self._get_target_position(); self.task.goal = desired_goal
        obs_dict = {"observation": robot_obs_vector.flatten().astype(np.float32),"achieved_goal": achieved_goal.astype(np.float32),"desired_goal": desired_goal.astype(np.float32)}
        return obs_dict

    def _get_info(self):
        """Return step info dictionary."""
        # ... (Implementation remains the same) ...
        achieved_pos = self.task.get_achieved_goal(); target_pos = self._get_target_position(); distance_error = np.linalg.norm(achieved_pos - target_pos) if achieved_pos is not None and target_pos is not None else float('inf'); is_success = self.task.is_success(achieved_pos, target_pos)
        return {"actual_ee_pos": achieved_pos if achieved_pos is not None else np.zeros(3),"reference_ee_pos": target_pos if target_pos is not None else np.zeros(3),"distance_error": distance_error, "sim_time": self.sim_time, "is_success": is_success }

    def reset(self, seed=None, options=None):
        """Reset environment and add trajectory visualization."""
        try:
            # print("DEBUG: Env reset called.") # Debug Print
            super().reset(seed=seed)
            self.sim_time = 0.0
            self.current_step = 0
            self.task.reset() # Resets task time and its internal goal state

            # --- Visualization Setup on Reset (if human mode) ---
            if self.render_mode == "human" and PYBULLET_AVAILABLE:
                # print("DEBUG: Resetting visualizations.") # Debug Print
                self._remove_visualizations() # Clean up previous episode's visuals
                self._draw_trajectory_line()
                self._create_target_marker()
                # Update marker to initial position AFTER getting initial info
                initial_target_pos = self._get_target_position() # Target at t=0
                self._update_target_marker(initial_target_pos)
            # ----------------------------------------------------

            # Get initial obs/info AFTER potentially moving marker
            observation = self._get_obs()
            info = self._get_info()

            return observation, info
        except Exception as e:
            # ... (Error handling) ...
            print("\n*** Error during PandaSinusoidTrackEnv reset ***"); traceback.print_exc(); print("*"*47+"\n")
            if hasattr(self, 'observation_space'): dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}; return dummy_obs, {}
            else: raise e


    def step(self, action):
        """Step environment, update target visualization."""
        terminated = False
        truncated = False
        try:
            self.robot.set_action(action)
            self.sim.step() # This steps the simulation N substeps
            self.sim_time += 1.0 / self.control_frequency # Increment time
            self.current_step += 1

            observation = self._get_obs() # Get new state dict
            info = self._get_info() # Get new info dict

            # --- Update Visualization ---
            if self.render_mode == "human" and PYBULLET_AVAILABLE:
                self._update_target_marker(info["reference_ee_pos"]) # Update marker to current ref pos
            # --------------------------

            achieved_goal = info["actual_ee_pos"]
            desired_goal = info["reference_ee_pos"]
            reward = self.task.compute_reward(achieved_goal, desired_goal, info)
            terminated = info["is_success"] # Always False for this task
            truncated = self.current_step >= MAX_EPISODE_STEPS

            # Explicit render call IS LIKELY NOT NEEDED for human mode
            # self.render() # Commenting this out - PyBullet GUI updates during sim.step()

            return observation, reward, terminated, truncated, info

        except Exception as e:
            # ... (Error handling) ...
             print(f"\n*** Error during step {self.current_step} ***"); traceback.print_exc(); print("*"*47+"\n")
             dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
             return dummy_obs, -100.0, False, True, {"is_success": False}

    # --- Visualization Helper Methods ---
    def _remove_visualizations(self):
        """Remove previous debug items."""
        if not PYBULLET_AVAILABLE: return
        # print("DEBUG: Removing old visualizations.") # Debug
        client_id = self.sim.physics_client
        for line_id in self.trajectory_line_ids:
            try: p.removeUserDebugItem(line_id, physicsClientId=client_id)
            except: pass # Ignore errors if item doesn't exist
        self.trajectory_line_ids = []
        if self.target_marker_id is not None:
            try: p.removeBody(self.target_marker_id, physicsClientId=client_id)
            except: pass # Ignore errors
            self.target_marker_id = None

    def _draw_trajectory_line(self, num_points=100, color=[1, 0.8, 0], width=2): # Orange color
        """Draw the reference trajectory as a series of connected lines."""
        if not PYBULLET_AVAILABLE: return
        # print(f"DEBUG: Drawing trajectory line ({num_points} points).") # Debug
        client_id = self.sim.physics_client
        points = [get_sinusoid_trajectory_point(t) for t in np.linspace(0, TRAJ_DURATION, num_points)]
        for i in range(len(points) - 1):
            try:
                line_id = p.addUserDebugLine( points[i], points[i+1], lineColorRGB=color,
                    lineWidth=width, physicsClientId=client_id )
                self.trajectory_line_ids.append(line_id)
            except Exception as e:
                 print(f"Error adding debug line segment {i}: {e}")
                 break # Stop drawing if there's an issue

    def _create_target_marker(self, radius=0.02, color=[1, 0.8, 0, 0.7]): # Orange, slightly transparent
        """Create a visual sphere marker for the current target."""
        if not PYBULLET_AVAILABLE: return
        # print("DEBUG: Creating target marker.") # Debug
        client_id = self.sim.physics_client
        try:
            # If marker already exists, remove it first (should be handled by _remove_visualizations)
            if self.target_marker_id is not None:
                 p.removeBody(self.target_marker_id, physicsClientId=client_id)

            visual_shape_id = p.createVisualShape( shapeType=p.GEOM_SPHERE, radius=radius,
                rgbaColor=color, physicsClientId=client_id )
            self.target_marker_id = p.createMultiBody( baseVisualShapeIndex=visual_shape_id,
                baseCollisionShapeIndex=-1, basePosition=[0,0,-1], # Start hidden
                useMaximalCoordinates=False, physicsClientId=client_id )
        except Exception as e:
            print(f"Error creating target marker: {e}")
            self.target_marker_id = None

    def _update_target_marker(self, target_position):
        """Update the position of the target marker sphere."""
        if self.target_marker_id is not None and PYBULLET_AVAILABLE:
            client_id = self.sim.physics_client
            try:
                p.resetBasePositionAndOrientation( self.target_marker_id, posObj=target_position,
                    ornObj=[0,0,0,1], physicsClientId=client_id )
            except Exception as e:
                 # Don't print every step maybe, or only if it repeats
                 # print(f"Error updating target marker position: {e}")
                 pass # Avoid spamming console if update fails

    # Override render to ensure it's available if needed
    def render(self):
        # In human mode, PyBullet handles rendering during sim step.
        # In rgb_array mode, this gets the image.
        if hasattr(self, 'sim') and self.sim is not None:
            return self.sim.render()
        return None

    def close(self):
        # Clean up visualizations before closing sim
        if self.render_mode == "human":
            self._remove_visualizations()
        if hasattr(self, 'sim') and self.sim is not None:
             self.sim.close()

# --- Registration ---
# ... (Registration code remains the same) ...
try:
    gym.register( id='PandaSinusoidTrack-v0', entry_point='custom_panda_env:PandaSinusoidTrackEnv',
        max_episode_steps=MAX_EPISODE_STEPS )
    print("Custom Environment 'PandaSinusoidTrack-v0' registered.")
except Exception as e: print(f"Error registering environment: {e}")