# replay_buffer.py
import numpy as np
import torch
import config # Import hyperparameters

class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.size = max_size
        self.counter = 0
        # Ensure input_shape is a tuple
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self.n_actions = n_actions

        self.states = np.zeros((max_size, *self.input_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.new_states = np.zeros((max_size, *self.input_shape), dtype=np.float32)
        self.terminals = np.zeros(max_size, dtype=bool) # Use bool for dones/terminals

        # PER specific
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_priority = 1.0 # Initial max priority
        self.alpha = config.PER_ALPHA
        self.beta = config.PER_BETA_START
        # Ensure PER_BETA_FRAMES is positive to avoid division by zero
        if config.PER_BETA_FRAMES > 0:
             self.beta_increment = (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES
        else:
             self.beta_increment = 0 # Beta will not anneal if frames is 0

    def store_transition(self, state, action, reward, new_state, done):
        index = self.counter % self.size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.terminals[index] = done
        # Add with max priority - ensures new samples are likely to be trained on soon
        self.priorities[index] = self.max_priority

        self.counter += 1

    def _get_probabilities(self):
        current_size = self.current_buffer_size()
        if current_size == 0:
            return np.array([], dtype=np.float32)
        # Use priorities only up to the current buffer size
        priorities_to_scale = self.priorities[:current_size]
        scaled_priorities = np.power(priorities_to_scale, self.alpha)
        sum_scaled_priorities = np.sum(scaled_priorities)
        if sum_scaled_priorities == 0:
             # Avoid division by zero if all priorities somehow became 0
             # Return uniform probability in this edge case
             print("Warning: Sum of scaled priorities is zero. Using uniform sampling.")
             return np.full(current_size, 1.0 / current_size, dtype=np.float32)
        return scaled_priorities / sum_scaled_priorities

    def _get_importance_sampling_weights(self, probabilities, indices):
        current_size = self.current_buffer_size()
        if current_size == 0 or len(indices) == 0:
            return np.array([], dtype=np.float32)

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calculate weights for the sampled indices
        probs_for_indices = probabilities[indices]
        # Avoid division by zero or NaNs if probabilities are zero
        probs_for_indices = np.maximum(probs_for_indices, 1e-9) # Add small epsilon

        weights = np.power(current_size * probs_for_indices, -self.beta)

        # Normalize weights by the maximum weight (max_weight = (N * min_prob)^-beta)
        min_prob = np.min(probabilities)
        min_prob = max(min_prob, 1e-9) # Avoid zero probability
        max_weight = np.power(current_size * min_prob, -self.beta)

        # Avoid division by zero if max_weight is zero (shouldn't happen with epsilon)
        if max_weight > 0:
            weights /= max_weight
        else:
             print("Warning: max_weight calculation resulted in zero.")
             weights.fill(1.0) # Fallback to weight 1

        return weights

    def sample(self, batch_size):
        max_mem = self.current_buffer_size()
        if max_mem < batch_size: # Cannot sample if buffer doesn't have enough elements
             # print(f"Warning: Buffer size ({max_mem}) is less than batch size ({batch_size}). Cannot sample.")
             return None # Return None to indicate sampling failed

        probabilities = self._get_probabilities()
        if len(probabilities) == 0: # Check if probabilities calculation failed
             return None

        # Ensure probabilities sum to 1, handle potential floating point inaccuracies
        probabilities /= np.sum(probabilities)

        indices = np.random.choice(max_mem, batch_size, p=probabilities, replace=True) # Sample with replacement

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        new_states = self.new_states[indices]
        dones = self.terminals[indices] # Use 'dones' or 'terminals' consistently

        weights = self._get_importance_sampling_weights(probabilities, indices)

        # Convert to PyTorch tensors on the correct device
        states_t = torch.tensor(states, dtype=torch.float32).to(config.DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(config.DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(config.DEVICE)
        new_states_t = torch.tensor(new_states, dtype=torch.float32).to(config.DEVICE)
        # Use bool or float for dones based on how loss calculation handles it
        # Using bool is generally cleaner if loss function supports it
        dones_t = torch.tensor(dones, dtype=torch.bool).to(config.DEVICE)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)

        return states_t, actions_t, rewards_t, new_states_t, dones_t, indices, weights_t

    def update_priorities(self, indices, td_errors):
        """Update priorities of sampled transitions."""
        # Ensure td_errors is numpy array
        priorities = np.abs(td_errors) + config.PER_EPSILON
        # priorities = np.abs(td_errors) + config.PER_EPSILON # Add epsilon to ensure non-zero priority
        clipped_priorities = np.minimum(priorities, self.max_priority) # Optional: clip priorities

        # Ensure indices are within the bounds of the current priorities array size
        valid_mask = indices < self.current_buffer_size()
        valid_indices = indices[valid_mask]
        valid_clipped_priorities = clipped_priorities[valid_mask]


        if len(valid_indices) > 0:
             self.priorities[valid_indices] = valid_clipped_priorities
             # Update max priority seen so far only based on valid updates
             current_max = np.max(valid_clipped_priorities)
             if current_max > self.max_priority:
                 self.max_priority = current_max
             # Alternative: self.max_priority = max(self.max_priority, np.max(valid_clipped_priorities))


    def current_buffer_size(self):
        """Returns the current number of items in the buffer."""
        return min(self.size, self.counter)

    def __len__(self):
        return self.current_buffer_size()