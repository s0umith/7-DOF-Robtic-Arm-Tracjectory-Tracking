# evaluate.py
import os
# Set env var BEFORE importing libraries that might use OpenMP (like numpy, torch)
# Set KMP_DUPLICATE_LIB_OK=TRUE to suppress OpenMP duplicate runtime errors (use with caution)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import gymnasium as gym
import numpy as np
import imageio # For saving GIFs
import time
import traceback
import matplotlib.pyplot as plt # Import matplotlib
import torch # Often needed indirectly or for checks

import config
# Make sure custom env is imported to register it
import custom_panda_env
from ddpg_agent import DDPGAgent

# --- Preprocessing Function --- (Must be defined here or imported from a utils file)
def preprocess_observation(obs_dict):
    """Concatenates components of the observation dictionary into a flat vector."""
    if not isinstance(obs_dict, dict):
        print(f"Warning: preprocess_observation expected dict, got {type(obs_dict)}")
        if isinstance(obs_dict, np.ndarray): return obs_dict.flatten().astype(np.float32)
        return None

    # Concatenate robot state ('observation') and target ('desired_goal')
    # Ensure this matches the input expected by the agent!
    obs = obs_dict.get('observation', np.array([]))
    desired = obs_dict.get('desired_goal', np.array([]))
    state_parts = [obs.flatten(), desired.flatten()]

    try:
        valid_parts = [part for part in state_parts if isinstance(part, np.ndarray) and part.size > 0]
        if not valid_parts:
            print("Warning: No valid observation parts found in preprocess_observation.")
            return None
        flat_obs = np.concatenate(valid_parts)
        return flat_obs.astype(np.float32)
    except ValueError as e:
        print(f"Error during concatenation in preprocess_observation: {e}")
        print(f"Shapes: {[p.shape for p in state_parts if isinstance(p, np.ndarray)]}")
        return None


def evaluate_agent(agent, env_name, n_episodes=10, render_mode="human", delay=0.03):
    """
    Evaluates the agent on the trajectory tracking task, calculates MSE,
    and generates trajectory, error, and action plots for each episode.
    """
    print(f"\n--- Starting Evaluation ({render_mode} mode) ---")
    if render_mode not in ["human", "rgb_array", "none"]:
        print(f"Warning: Invalid render_mode '{render_mode}'. Defaulting to 'none'.")
        render_mode = "none"

    env = None
    n_actions = None # To store action dimensions
    try:
        # Use appropriate render mode, handle 'none' case needing a valid mode for creation
        # For 'none' mode, still create with 'rgb_array' internally to satisfy potential init requirements
        create_render_mode = render_mode if render_mode != "none" else "rgb_array"
        print(f"Creating evaluation environment '{env_name}' with creation mode '{create_render_mode}'...")
        env = gym.make(env_name, render_mode=create_render_mode)
        n_actions = env.action_space.shape[0] # Get action dimensions
        print("Evaluation environment created.")
    except Exception as e:
        print(f"Error creating evaluation environment '{env_name}': {e}")
        traceback.print_exc()
        return None # Cannot evaluate if env creation fails

    agent.prep_for_eval() # Set agent networks to evaluation mode

    total_rewards = []
    episode_lengths = []
    all_episode_mse = []
    frames = [] # Only used if render_mode == "rgb_array"

    # --- Create a directory for plots ---
    plot_dir = f"eval_plots_{config.MODEL_NAME}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving evaluation plots to: {plot_dir}")

    for i in range(n_episodes):
        print(f"--- Starting Eval Episode {i+1}/{n_episodes} ---")
        try:
            observation_dict, info = env.reset() # Env reset adds trajectory visuals in human mode now
            state = preprocess_observation(observation_dict)
            if state is None:
                print(f"Warning: Failed preprocess initial observation for eval ep {i+1}. Skipping.")
                continue
        except Exception as e:
             print(f"Error during env.reset() for eval ep {i+1}: {e}")
             traceback.print_exc()
             continue # Skip episode


        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        # Store positions and actions for plotting
        actual_positions = []
        ref_positions = []
        actions_history = [] # <-- Store actions history
        last_action = np.zeros(n_actions) # For optional smoothing

        # --- Initial Frame for GIF ---
        if render_mode == "rgb_array":
             try:
                 # Render AFTER reset, which might draw initial state
                 frame = env.render()
                 if frame is not None: frames.append(frame)
                 else: print(f"Warning: env.render() returned None at start of eval ep {i+1}.")
             except Exception as e: print(f"Error rendering initial frame ep {i+1}: {e}")

        # --- Step Loop ---
        while not (terminated or truncated):
            if state is None: # Safety check
                print(f"Error: current_state is None in eval ep {i+1}. Terminating.")
                break

            raw_action = agent.choose_action(state, evaluate=True) # Get policy output

            # --- Optional Action Smoothing ---
            # Uncomment below to apply smoothing
            smoothing_factor = 0.8 # Tune this (0=none, 0.9=heavy)
            smoothed_action = smoothing_factor * last_action + (1.0 - smoothing_factor) * raw_action
            action_to_send = smoothed_action
            last_action = action_to_send # Update last action for next step
            # ---------------------------------

            # --- Use raw action if not smoothing ---
            # action_to_send = raw_action # Comment this out if using smoothing

            actions_history.append(action_to_send) # <-- Store the action *sent* to env

            try:
                next_observation_dict, reward, terminated, truncated, info = env.step(action_to_send)
                # Combine done flags
                done = terminated or truncated
                next_state = preprocess_observation(next_observation_dict)
                # Max steps truncation is handled by gym.register's max_episode_steps
            except Exception as e:
                print(f"Error during env.step() at step {steps+1} in eval ep {i+1}: {e}")
                traceback.print_exc()
                truncated = True
                next_state = None # Stop processing if step fails badly
                reward = -10.0

            # --- Store positions from info dict ---
            actual_pos = info.get('actual_ee_pos')
            ref_pos = info.get('reference_ee_pos')
            if actual_pos is not None and ref_pos is not None:
                actual_positions.append(actual_pos)
                ref_positions.append(ref_pos)
            elif steps == 0: print(f"Warning: Position info missing from env info dict in ep {i+1}.")

            # --- Rendering / Delay ---
            if render_mode == "human":
                # For human mode, PyBullet might update view during env.step() or env.render()
                # A small sleep is needed to make it watchable.
                time.sleep(delay)
                # Explicit render call might not be needed, but can force update if view lags
                # env.render()
            elif render_mode == "rgb_array":
                 try:
                     frame = env.render() # Get frame for GIF
                     if frame is not None: frames.append(frame)
                 except Exception as e: print(f"Error rendering frame during step {steps+1} ep {i+1}: {e}")

            state = next_state
            episode_reward += reward
            steps += 1

            # Check combined done flag
            if done: break
            # Safety break if somehow loop continues past max steps
            if steps >= config.MAX_EPISODE_STEPS * 1.1 : # Allow slight overshoot
                 print(f"Warning: Episode {i+1} manually truncated due to excessive steps ({steps}).")
                 truncated = True; break


        # --- End of Episode Calculations & PLOTTING ---
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        episode_mse = float('nan')
        can_plot = False
        # Ensure we have enough data points for meaningful plots/calculations
        if actual_positions and ref_positions and len(actual_positions) > 1:
             actual_positions_np = np.array(actual_positions)
             ref_positions_np = np.array(ref_positions)
             actions_np = np.array(actions_history) # Convert actions too

             # Validate shapes before proceeding
             if actual_positions_np.shape == ref_positions_np.shape and \
                actual_positions_np.ndim == 2 and actual_positions_np.shape[1] == 3 and \
                actions_np.ndim == 2 and actions_np.shape[0] == actual_positions_np.shape[0] and \
                actions_np.shape[1] == n_actions:

                 squared_errors = np.sum((actual_positions_np - ref_positions_np)**2, axis=1)
                 episode_mse = np.mean(squared_errors) if squared_errors.size > 0 else 0
                 can_plot = True
                 print(f"Finished Eval Episode {i+1}/{n_episodes} | Score: {episode_reward:.2f} | Steps: {steps} | MSE: {episode_mse:.6f}")
             else:
                 print(f"Finished Eval Episode {i+1}/{n_episodes} | Score: {episode_reward:.2f} | Steps: {steps} | MSE: N/A (Shape mismatch)")
                 print(f"    Actual Pos Shape: {actual_positions_np.shape}, Ref Pos Shape: {ref_positions_np.shape}, Actions Shape: {actions_np.shape}")
        else:
             print(f"Finished Eval Episode {i+1}/{n_episodes} | Score: {episode_reward:.2f} | Steps: {steps} | MSE: N/A (No/Insufficient position data)")

        all_episode_mse.append(episode_mse)

        # --- Generate Plots if data is valid ---
        if can_plot:
             try:
                 time_steps = np.arange(len(actual_positions_np)) * (1.0 / config.CONTROL_FREQ)

                 # --- Trajectory Plot ---
                 fig_traj, axs_traj = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                 labels = ['X', 'Y', 'Z']; colors = ['r', 'g', 'b']
                 for dim in range(3):
                    axs_traj[dim].plot(time_steps, ref_positions_np[:, dim], linestyle='--', color=colors[dim], label=f'Reference {labels[dim]}')
                    axs_traj[dim].plot(time_steps, actual_positions_np[:, dim], linestyle='-', color=colors[dim], label=f'Actual {labels[dim]}')
                    axs_traj[dim].set_ylabel(f'{labels[dim]} Position (m)'); axs_traj[dim].legend(loc='upper right'); axs_traj[dim].grid(True)
                 axs_traj[2].set_xlabel('Time (s)')
                 fig_traj.suptitle(f'Trajectory Tracking - Episode {i+1} (MSE: {episode_mse:.6f})'); fig_traj.tight_layout(rect=[0, 0.03, 1, 0.97])
                 plot_filename_traj = os.path.join(plot_dir, f"eval_ep_{i+1:02d}_trajectory.png")
                 fig_traj.savefig(plot_filename_traj); plt.close(fig_traj)

                 # --- Error Plot ---
                 error_magnitude = np.linalg.norm(actual_positions_np - ref_positions_np, axis=1)
                 fig_err, ax_err = plt.subplots(1, 1, figsize=(10, 4))
                 ax_err.plot(time_steps, error_magnitude, label='Tracking Error Magnitude', color='purple')
                 ax_err.set_xlabel('Time (s)'); ax_err.set_ylabel('Error Magnitude (m)'); ax_err.set_title(f'Tracking Error - Episode {i+1}')
                 ax_err.legend(); ax_err.grid(True); ax_err.set_ylim(bottom=0); fig_err.tight_layout()
                 plot_filename_err = os.path.join(plot_dir, f"eval_ep_{i+1:02d}_error.png")
                 fig_err.savefig(plot_filename_err); plt.close(fig_err)

                 # --- Action Plot ---
                 fig_act, axs_act = plt.subplots(n_actions, 1, figsize=(10, 2 * n_actions), sharex=True)
                 if n_actions == 1: axs_act = [axs_act] # Handle single action dim case
                 for act_dim in range(n_actions):
                     axs_act[act_dim].plot(time_steps, actions_np[:, act_dim], label=f'Action Dim {act_dim+1}')
                     axs_act[act_dim].set_ylabel(f'Action[{act_dim}]')
                     axs_act[act_dim].grid(True)
                     if hasattr(env, 'action_space'): # Add limits if possible
                         axs_act[act_dim].axhline(env.action_space.low[act_dim], color='gray', linestyle=':', linewidth=0.8)
                         axs_act[act_dim].axhline(env.action_space.high[act_dim], color='gray', linestyle=':', linewidth=0.8)
                     axs_act[act_dim].legend(loc='upper right')
                 axs_act[-1].set_xlabel('Time (s)')
                 fig_act.suptitle(f'Control Actions - Episode {i+1}')
                 fig_act.tight_layout(rect=[0, 0.03, 1, 0.97])
                 plot_filename_act = os.path.join(plot_dir, f"eval_ep_{i+1:02d}_actions.png")
                 fig_act.savefig(plot_filename_act); plt.close(fig_act)

                 print(f"Plots saved for episode {i+1}.")

             except Exception as plot_e:
                 print(f"Error generating plots for episode {i+1}: {plot_e}")
                 traceback.print_exc()


    # --- Final Analytics ---
    if env: env.close()
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    avg_mse = np.nanmean(all_episode_mse) if all_episode_mse else float('nan') # Use nanmean to ignore NaNs

    print("\n--- Evaluation Summary ---")
    print(f"Episodes Evaluated: {n_episodes}")
    print(f"Average Score: {avg_reward:.2f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Plots saved in: {plot_dir}")
    print("-------------------------")

    # --- Save GIF ---
    if render_mode == "rgb_array" and frames:
        gif_filename = config.RENDER_FILENAME
        print(f"Saving animation to {gif_filename}...")
        try:
            duration = max(0.01, config.RENDER_DELAY / 1000.0)
            imageio.mimsave(gif_filename, frames, duration=duration)
            print("Animation saved.")
        except Exception as e: print(f"Error saving GIF: {e}")
    elif render_mode == "rgb_array" and not frames:
        print("Warning: 'rgb_array' mode selected, but no frames were captured for GIF.")

    metrics = {"avg_score": avg_reward, "avg_mse": avg_mse, "avg_length": avg_length}
    return metrics

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CHOOSE EVALUATION MODE ---
    EVAL_RENDER_MODE = "human"     # For live PyBullet view + plots
    # EVAL_RENDER_MODE = "rgb_array" # For saving a GIF + plots
    # EVAL_RENDER_MODE = "none"      # For fastest evaluation + plots
    env_name = config.ENV_NAME

    # --- Agent Initialization ---
    init_env = None; agent = None
    try:
        print("Creating environment instance for agent initialization...")
        init_env = gym.make(env_name, render_mode="rgb_array") # Use rgb_array for init
        agent = DDPGAgent(env=init_env)
        print("Agent initialized.")
    except Exception as e: print(f"Fatal Error init: {e}"); traceback.print_exc()
    finally:
         if init_env: init_env.close() # Close the temporary env
    if agent is None: exit() # Exit if agent init failed

    # --- Load Models and Run Evaluation ---
    try:
        print(f"Loading trained models for {config.MODEL_NAME}...")
        agent.load_models()
        print("Models loaded successfully.")
        # Run evaluation and plotting
        evaluate_agent(agent=agent,
                       env_name=env_name,
                       n_episodes=config.EVAL_EPISODES,
                       render_mode=EVAL_RENDER_MODE,
                       delay=config.RENDER_DELAY / 1000.0) # Convert config delay (ms) to s
    except FileNotFoundError:
        print(f"\nError: Could not load pre-trained models. Checkpoint files not found in:")
        print(os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
        print("Please ensure the agent was trained and models were saved.")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        traceback.print_exc()