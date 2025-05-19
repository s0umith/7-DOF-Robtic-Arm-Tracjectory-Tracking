# networks.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config # Import hyperparameters

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name="actor", model_name=config.MODEL_NAME, checkpoints_dir=config.CHECKPOINT_DIR):
        super(ActorNetwork, self).__init__()
        self.checkpoints_file = os.path.join(checkpoints_dir, model_name, name + ".pth")
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)

        # Adjusted layer sizes based on common practices for continuous control
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, n_actions) # Output layer for actions

        # Weight initialization (optional but can help)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.uniform_(self.pi.weight, -3e-3, 3e-3) # Small weights for output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use tanh to bound actions between -1 and 1 (assuming env actions are scaled)
        pi = torch.tanh(self.pi(x))
        return pi

    def save_checkpoint(self):
        print(f"... saving checkpoint '{self.checkpoints_file}' ...")
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        print(f"... loading checkpoint '{self.checkpoints_file}' ...")
        self.load_state_dict(torch.load(self.checkpoints_file, map_location=config.DEVICE))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name="critic", model_name=config.MODEL_NAME, checkpoints_dir=config.CHECKPOINT_DIR):
        super(CriticNetwork, self).__init__()
        self.checkpoints_file = os.path.join(checkpoints_dir, model_name, name + ".pth")
        os.makedirs(os.path.dirname(self.checkpoints_file), exist_ok=True)

        # Input to first layer includes state and action dimensions
        self.fc1 = nn.Linear(input_dims + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1) # Output layer for Q-value

        # Weight initialization
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.uniform_(self.q.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension (dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x) # Q-value is unbounded
        return q

    def save_checkpoint(self):
        print(f"... saving checkpoint '{self.checkpoints_file}' ...")
        torch.save(self.state_dict(), self.checkpoints_file)

    def load_checkpoint(self):
        print(f"... loading checkpoint '{self.checkpoints_file}' ...")
        self.load_state_dict(torch.load(self.checkpoints_file, map_location=config.DEVICE))