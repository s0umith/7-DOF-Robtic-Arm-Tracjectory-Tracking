# train.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import gymnasium as gym
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
import traceback
import time # <-- Import time module


# Assuming other necessary files are in the same directory or install path
import config
import custom_panda_env # Register custom environment when imported
from ddpg_agent import DDPGAgent
# Import the evaluation function - ensure evaluate.py is accessible
try:
    # Import evaluate_agent and ensure preprocess_observation is also accessible if needed elsewhere
    from evaluate import evaluate_agent, preprocess_observation
except ImportError:
    print("Warning: Could not import 'evaluate_agent' or 'preprocess_observation' from evaluate.py.")
    # Define preprocess_observation here ONLY IF evaluate.py cannot be imported AND it's needed elsewhere
    # It's generally better to fix the import.
    def preprocess_observation(obs_dict):
        if not isinstance(obs_dict, dict):
            print(f"Warning: preprocess_observation expected dict, got {type(obs_dict)}")
            if isinstance(obs_dict, np.ndarray): return obs_dict.flatten().astype(np.float32)
            return None
        obs = obs_dict.get('observation', np.array([])); desired = obs_dict.get('desired_goal', np.array([]))
        state_parts = [obs.flatten(), desired.flatten()]
        valid_parts = [part for part in state_parts if isinstance(part, np.ndarray) and part.size > 0]
        if not valid_parts: return None
        try: return np.concatenate(valid_parts).astype(np.float32)
        except ValueError as e: print(f"Error concatenating in preprocess: {e}"); return None
    # Define a dummy evaluate_agent if import fails, to prevent crashes during periodic eval
    def evaluate_agent(*args, **kwargs):
        print("Placeholder: evaluate_agent not imported. Periodic evaluation skipped.")
        return None

import torch
if __name__ == "__main__":
    # Set OpenMP environment variable workaround if needed (before torch/numpy imports)
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    print(f"Using device: {config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}") # Check torch version
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        # print(f"cuDNN version: {torch.backends.cudnn.version()}") # Might require cuDNN install check
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        # print(f"cuDNN enabled: {torch.backends.cudnn.enabled}") # Might require cuDNN install check

    # --- Create Environment FIRST ---
    env = None
    try:
        print(f"Creating training environment '{config.ENV_NAME}'...")
        # Use 'rgb_array' during creation to satisfy potential reqs, even if not rendering
        # You could use render_mode=None if environment creation works without it
        env = gym.make(config.ENV_NAME, render_mode="rgb_array")
        print("Training environment created.")
    except Exception as e:
         print(f"Fatal Error: Could not create environment '{config.ENV_NAME}'. Exiting.")
         print(f"Details: {e}")
         traceback.print_exc()
         if env: env.close()
         exit()

    # --- Initialize Agent using the created env ---
    agent = None
    try:
        print("Initializing agent...")
        agent = DDPGAgent(env=env) # Pass the created env instance
        print("Agent initialized.")
    except Exception as e:
         print(f"Fatal Error: Could not initialize agent. Exiting.")
         print(f"Details: {e}")
         traceback.print_exc()
         if env: env.close()
         exit()

    # --- Training Setup ---
    best_avg_score = -np.inf # Initialize best average score (higher reward is better, less negative)
    score_history = deque(maxlen=100) # Store last 100 scores for averaging
    avg_score_history = [] # Store average scores for plotting
    print(f"Starting training for {config.N_GAMES} episodes on {config.ENV_NAME}...")
    print(f"Max steps per episode: {config.MAX_EPISODE_STEPS}")
    print(f"Prioritized Replay Buffer Size: {config.BUFFER_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Warmup Steps (random actions): {config.WARMUP_BATCHES * config.BATCH_SIZE}")
    print(f"Periodic evaluation every {config.EVAL_FREQ} episodes.")
    print(f"Periodic timing info every 200 steps.") # Print timing less often

    total_steps = 0

    # --- Main Training Loop ---
    for i_episode in range(config.N_GAMES):
        start_time_episode = time.time() # Time the whole episode
        try:
            observation_dict, info = env.reset()
            current_state = preprocess_observation(observation_dict) # Preprocess
            if current_state is None:
                 print(f"Warning: Failed to preprocess initial observation for episode {i_episode+1}. Skipping.")
                 continue
        except Exception as e:
             print(f"Error during env.reset() or preprocessing for episode {i_episode+1}: {e}")
             traceback.print_exc()
             continue # Skip episode

        terminated = False
        truncated = False
        episode_score = 0
        episode_steps = 0

        # Accumulators for step timings within an episode
        time_action_total = 0.0
        time_step_total = 0.0
        time_remember_total = 0.0
        time_learn_total = 0.0
        learn_calls = 0

        while not (terminated or truncated):
            if current_state is None: # Safety check
                print(f"Error: current_state is None in episode {i_episode+1}. Terminating episode.")
                break

            # --- Time Action Selection ---
            time_action_start = time.time()
            if total_steps < config.WARMUP_BATCHES * config.BATCH_SIZE:
                 action = env.action_space.sample()
            else:
                # Ensure state has the correct dimension before passing to agent
                if current_state.shape[0] != agent.input_dims:
                    print(f"Error: State shape mismatch before action! Got {current_state.shape}, expected ({agent.input_dims},). Terminating episode.")
                    truncated = True # End the episode
                    continue # Skip rest of the loop iteration
                action = agent.choose_action(current_state, evaluate=False)
            time_action_end = time.time()
            time_action_total += (time_action_end - time_action_start)
            # -----------------------------

            next_state = None # Initialize next_state
            reward = 0.0 # Initialize reward

            # --- Time Environment Step ---
            time_step_start = time.time()
            try:
                next_observation_dict, reward, terminated, truncated, info = env.step(action)
                # Combine env flags with step limit
                truncated = truncated or (episode_steps + 1 >= config.MAX_EPISODE_STEPS)
                next_state = preprocess_observation(next_observation_dict) # Preprocess
                if next_state is None:
                     print(f"Warning: Preprocessing next_observation failed at step {episode_steps+1}, episode {i_episode+1}.")
                     truncated = True # End episode if state becomes invalid
                     # next_state remains None, will skip remember

            except Exception as e:
                print(f"Error during env.step() at step {episode_steps+1} in episode {i_episode+1}: {e}")
                traceback.print_exc()
                truncated = True # End episode on error
                next_state = None
                reward = -10.0 # Example penalty
            time_step_end = time.time()
            time_step_total += (time_step_end - time_step_start)
            # ---------------------------

            # --- Time Remember Step ---
            time_remember_start = time.time()
            # Store experience only if both states are valid
            if current_state is not None and next_state is not None:
                 # Ensure next_state also has correct dimension
                 if next_state.shape[0] == agent.input_dims:
                      agent.remember(current_state, action, reward, next_state, terminated or truncated) # Use combined flag
                 else:
                      print(f"Warning: Skipping remember at step {episode_steps+1}, ep {i_episode+1} due to invalid next_state shape {next_state.shape}")
                      truncated = True # End episode if state processing fails
            # Do not remember if next_state is invalid or current_state became invalid
            elif current_state is not None and next_state is None:
                 print(f"Info: Skipped remember at step {episode_steps+1}, ep {i_episode+1} due to invalid next_state.")
            # else: (current_state is None case already handled)

            time_remember_end = time.time()
            time_remember_total += (time_remember_end - time_remember_start)
            # -------------------------

            # --- Time Learn Step ---
            time_learn_start = time.time()
            learn_executed_this_step = False
            if total_steps >= config.WARMUP_BATCHES * config.BATCH_SIZE and len(agent.memory) >= config.BATCH_SIZE:
                 agent.learn()
                 learn_executed_this_step = True
                 learn_calls += 1
            time_learn_end = time.time()
            if learn_executed_this_step:
                 time_learn_total += (time_learn_end - time_learn_start)
            # -----------------------

            # Prepare for next iteration
            current_state = next_state
            episode_score += reward
            episode_steps += 1
            total_steps += 1

            # --- Periodic Timing Print ---
            if total_steps % 200 == 0 and episode_steps > 0: # Print less often
                 # Calculate average time per step for this episode so far
                 avg_act = time_action_total / episode_steps
                 avg_step = time_step_total / episode_steps
                 avg_rem = time_remember_total / episode_steps
                 avg_learn = (time_learn_total / learn_calls) if learn_calls > 0 else 0
                 print(f"      Timing Avg (Ep {i_episode+1}, first {episode_steps} steps): "
                       f"Act={avg_act:.5f}s | EnvStep={avg_step:.5f}s | Rem={avg_rem:.5f}s | Learn={avg_learn:.5f}s")
            # ---------------------------


        # --- End of Episode ---
        end_time_episode = time.time()
        episode_duration = end_time_episode - start_time_episode

        score_history.append(episode_score)
        avg_score = np.mean(score_history)
        avg_score_history.append(avg_score)

        # Save best model based on average score (higher is better)
        # Ensure history is populated before checking score
        if len(score_history) >= 100 and avg_score > best_avg_score :
            print(f"New best average score: {avg_score:.2f} (previous best: {best_avg_score:.2f})")
            best_avg_score = avg_score
            agent.save_models() # Save the improved model

        # Logging (Include episode duration)
        print(f"Ep {i_episode+1}/{config.N_GAMES} | Steps: {episode_steps} | Score: {episode_score:.2f} | "
              f"Avg Score: {avg_score:.2f} | Buffer: {len(agent.memory)} | Ep Time: {episode_duration:.2f}s")


        # --- Periodic Evaluation ---
        # Run evaluation every EVAL_FREQ episodes (and not on the very first episode)
        if (i_episode + 1) % config.EVAL_FREQ == 0 and i_episode > 0 :
             eval_episodes_periodic = 3 # Number of episodes for quick check
             print(f"\n--- Running Periodic Evaluation at Episode {i_episode+1} ({eval_episodes_periodic} episodes) ---")
             try:
                 # Use "none" for render_mode to avoid visuals slowing down training
                 eval_metrics = evaluate_agent(agent, config.ENV_NAME,
                                               n_episodes=eval_episodes_periodic,
                                               render_mode="none") # Change to "human" to watch if needed
                 if eval_metrics:
                     # Access MSE safely using .get with a default value (e.g., NaN) if key missing
                     avg_mse = eval_metrics.get('avg_mse', float('nan'))
                     print(f"--- Periodic Eval Result --- Avg Score: {eval_metrics['avg_score']:.2f}, Avg MSE: {avg_mse:.6f}")
                 else:
                     print("--- Periodic Eval Result --- Evaluation failed or returned None.")
                 print(f"--- Resuming Training ---\n")
             except NameError:
                  print("Periodic Evaluation skipped: 'evaluate_agent' not defined/imported.")
             except Exception as eval_e:
                 print(f"Error during periodic evaluation: {eval_e}")
                 traceback.print_exc()
        # --- END Periodic Evaluation Block ---


    # --- Cleanup and Plotting ---
    if env:
        env.close()
        print("Training environment closed.")
    print("Training finished.")

    # Plot average score history
    if avg_score_history:
        episodes = list(range(1, len(avg_score_history) + 1))
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, avg_score_history, label='Avg Score (Last 100 Episodes)', color='tab:blue')
        plt.xlabel("Episode")
        plt.ylabel("Average Score")
        plt.title(f"Training Progress: {config.MODEL_NAME} on {config.ENV_NAME}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f"{config.MODEL_NAME}_training_progress.png"
        try:
            plt.savefig(plot_filename)
            print(f"Training progress plot saved to {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        # plt.show() # Uncomment to display plot interactively
    else:
        print("No average score history to plot.")