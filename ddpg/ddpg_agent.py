# ddpg_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import gymnasium as gym # Import gym for spaces check

import config # Import hyperparameters
from networks import ActorNetwork, CriticNetwork
from replay_buffer import PrioritizedReplayBuffer

class DDPGAgent:
    def __init__(self, env):
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.policy_delay = config.POLICY_DELAY
        self.noise_stddev = config.NOISE_STDDEV
        self.noise_clip = config.NOISE_CLIP
        self.learn_step_counter = 0 # For policy delay

        self.env = env # Keep env reference if needed

        # --- Determine input_dims from the Dict observation space ---
        if not isinstance(env.observation_space, gym.spaces.Dict):
             raise ValueError(f"Agent expects a Dict observation space, but got {type(env.observation_space)}")

        # Calculate dims based on the keys USED in preprocess_observation
        # Ensure preprocess_observation function's logic matches this!
        # Assuming preprocess_observation uses 'observation' and 'desired_goal'
        try:
             # Get shape of the main robot observation vector
             obs_shape = env.observation_space['observation'].shape
             obs_dim = obs_shape[0] if obs_shape else 0 # Handle cases where shape might be empty tuple

             # Get shape of the desired goal vector
             goal_shape = env.observation_space['desired_goal'].shape
             goal_dim = goal_shape[0] if goal_shape else 0

             # Input dims = robot obs dim + desired goal dim (based on preprocess function)
             self.input_dims = obs_dim + goal_dim
             print(f"Agent calculated input_dims: {self.input_dims} (robot_obs={obs_dim}, desired_goal={goal_dim})")

             if self.input_dims == 0:
                  raise ValueError("Calculated input_dims is zero. Check observation space structure.")

        except KeyError as e:
             raise ValueError(f"Observation space Dict missing expected key used for input_dims calculation: {e}. Available keys: {list(env.observation_space.spaces.keys())}")
        except AttributeError as e:
             raise ValueError(f"Error accessing shape from observation space components: {e}")
        except Exception as e:
             raise ValueError(f"Unexpected error calculating input_dims: {e}")


        self.n_actions = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        if np.any(np.abs(self.min_action + self.max_action) > 1e-6):
             print("Warning: Action space may not be symmetric. Scaling might be inaccurate.")
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        # Handle case where action range is zero
        if self.action_scale == 0:
             print("Warning: Action scale is zero. Actions will not be scaled.")
             self.action_scale = 1.0 # Avoid division by zero, use action directly
             self.action_bias = 0.0


        # --- Initialize Buffer and Networks ---
        self.memory = PrioritizedReplayBuffer(config.BUFFER_SIZE, self.input_dims, self.n_actions)
        self._initialize_networks()


    def _initialize_networks(self):
        # Pass correct model name from config
        model_name = config.MODEL_NAME
        self.actor = ActorNetwork(self.input_dims, self.n_actions, name="actor", model_name=model_name).to(config.DEVICE)
        self.critic = CriticNetwork(self.input_dims, self.n_actions, name="critic", model_name=model_name).to(config.DEVICE)
        self.target_actor = ActorNetwork(self.input_dims, self.n_actions, name="target_actor", model_name=model_name).to(config.DEVICE)
        self.target_critic = CriticNetwork(self.input_dims, self.n_actions, name="target_critic", model_name=model_name).to(config.DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR, weight_decay=config.OPTIMIZER_WEIGHT_DECAY)

        self.update_parameters(tau=1.0) # Initialize target networks
        self.prep_for_eval() # Set target nets to eval mode initially

    def choose_action(self, state, evaluate=False):
        """Chooses an action based on the FLAT state vector."""
        if not isinstance(state, np.ndarray):
             state = np.array(state, dtype=np.float32)
        if state.ndim > 1: state = state.flatten() # Ensure flat
        if state.shape[0] != self.input_dims:
            print(f"Warning: choose_action received state with wrong shape {state.shape}, expected ({self.input_dims},). Agent might fail.")
            # Handle error? Return zero action? For now, proceed but warn.
            # Pad or truncate? Risky. Better to fix state source.

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        self.actor.eval()
        with torch.no_grad():
             mu = self.actor(state_tensor) # Action in [-1, 1]
        self.actor.train()

        if not evaluate:
            noise = torch.normal(mean=0.0, std=self.noise_stddev, size=mu.shape).to(config.DEVICE)
            # noise = torch.clamp(noise, -self.noise_clip, self.noise_clip) # Optional noise clipping
            mu = mu + noise
            mu = torch.clamp(mu, -1.0, 1.0) # Clip after adding noise

        # Scale and clip action
        action_scaled = mu.squeeze(0).cpu().numpy() * self.action_scale + self.action_bias
        action_clipped = np.clip(action_scaled, self.min_action, self.max_action)
        return action_clipped

    def remember(self, state, action, reward, new_state, done):
        """Stores transition with FLAT states and UNSCALED action."""
        # Unscale action back to [-1, 1] range
        if not isinstance(action, np.ndarray): action = np.array(action)
        if self.action_scale != 0:
             unscaled_action = np.clip((action - self.action_bias) / self.action_scale, -1.0, 1.0)
        else:
             unscaled_action = np.clip(action, -1.0, 1.0)

        # Ensure states are flat numpy arrays
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        if not isinstance(new_state, np.ndarray): new_state = np.array(new_state, dtype=np.float32)
        if state.ndim > 1: state = state.flatten()
        if new_state.ndim > 1: new_state = new_state.flatten()

        # Add checks for state dimension consistency
        if state.shape[0] != self.input_dims or new_state.shape[0] != self.input_dims:
             print(f"Error in remember: State shapes incorrect! State: {state.shape}, New State: {new_state.shape}, Expected: ({self.input_dims},)")
             # Optionally skip remembering this transition
             return

        self.memory.store_transition(state, unscaled_action, reward, new_state, done)

    def learn(self):
        if len(self.memory) < config.BATCH_SIZE: # Use config directly
            return

        sample_result = self.memory.sample(config.BATCH_SIZE)
        if sample_result is None:
            return

        states, actions, rewards, new_states, dones, indices, weights = sample_result

        rewards = rewards.unsqueeze(1)
        dones_float = dones.unsqueeze(1).type(torch.float32) # For calculation
        weights = weights.unsqueeze(1)

        # --- Critic Update ---
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

        with torch.no_grad():
            noise = torch.normal(mean=0.0, std=self.noise_stddev, size=actions.shape).to(config.DEVICE)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(self.target_actor(new_states) + noise, -1.0, 1.0)
            target_q = rewards + (self.gamma * self.target_critic(new_states, next_actions) * (1.0 - dones_float))

        current_q = self.critic(states, actions)
        td_errors = (target_q - current_q).abs()
        critic_loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.squeeze(1).detach().cpu().numpy()) # Pass numpy errors

        # --- Delayed Actor Update ---
        self.learn_step_counter += 1
        if self.learn_step_counter % self.policy_delay == 0:
            self.actor.train()
            for param in self.critic.parameters(): param.requires_grad = False # Freeze critic

            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters(): param.requires_grad = True # Unfreeze critic

            # --- Delayed Target Updates ---
            self.update_parameters()

    def update_parameters(self, tau=None):
        if tau is None: tau = self.tau
        # Soft update critic
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # Soft update actor
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        print(f"---- saving models for {config.MODEL_NAME} ----")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print(f"---- loading models for {config.MODEL_NAME} ----")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.prep_for_eval() # Ensure eval mode after loading

    def prep_for_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()