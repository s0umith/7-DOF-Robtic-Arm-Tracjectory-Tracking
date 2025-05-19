# her.py
import numpy as np
import config # Import hyperparameters

def her_augmentation(replay_buffer, episode_transitions, env):
    """
    Augments the replay buffer with HER using the 'future' strategy.
    Stores augmented transitions with maximum priority.

    Args:
        replay_buffer (PrioritizedReplayBuffer): The buffer to store transitions in.
        episode_transitions (list): A list of tuples, where each tuple is
                                    (observation, action, reward, next_observation, done).
                                    Observations are dictionaries.
        env: The Gym environment instance (needed for compute_reward).
    """
    k = config.HER_K
    num_transitions = len(episode_transitions)

    for t in range(num_transitions):
        observation, action, _, next_observation, _ = episode_transitions[t]
        obs_t = observation['observation']
        achieved_goal_t = observation['achieved_goal']
        # desired_goal_t = observation['desired_goal'] # Original desired goal
        next_obs_tp1 = next_observation['observation']
        achieved_goal_tp1 = next_observation['achieved_goal']

        # Sample k future goals from the same trajectory
        future_indices = np.random.randint(t, num_transitions, size=k)

        for future_idx in future_indices:
            # Get the achieved goal from the future state as the new desired goal
            _, _, _, future_next_obs, _ = episode_transitions[future_idx]
            future_achieved_goal = future_next_obs['achieved_goal']
            new_desired_goal = future_achieved_goal # Relabel

            # Recalculate reward based on the new goal
            # Note: compute_reward often takes achieved_goal, desired_goal, info
            # Panda-gym might just need achieved_goal, desired_goal
            # Adjust the call based on your specific env.compute_reward signature.
            # Often info={} is sufficient if not used.
            info = {} # Placeholder info if needed by compute_reward
            # We use achieved_goal_tp1 (the goal achieved *after* taking the action)
            # and the new_desired_goal for the reward calculation.
            # Use env.unwrapped to access the method from the original environment
            reward = env.unwrapped.compute_reward(achieved_goal_tp1, new_desired_goal, info) # CORRECTED LINE

            # Determine if the new goal was achieved (for 'done' signal in HER)
            # This depends on how your env calculates success. Often distance based.
            # Using distance threshold is common. Panda-Gym's reward is often -0.0 or -1.0
            # If reward is 0 (or close), goal is achieved.
            goal_achieved_threshold = -1e-6 # Check if reward is not the failure reward (-1)
            done = (reward > goal_achieved_threshold)


            # Construct the new state and next_state with the relabeled goal
            state_new = np.concatenate([obs_t, achieved_goal_t, new_desired_goal])
            next_state_new = np.concatenate([next_obs_tp1, achieved_goal_tp1, new_desired_goal])

            # Store the augmented transition in the prioritized buffer
            # It gets the current max priority
            replay_buffer.store_transition(state_new, action, reward, next_state_new, done)