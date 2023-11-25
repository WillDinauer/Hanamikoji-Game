import HanamikojiEnvironment
import ReinforcementLearningAgent
import random
import RandomPlayer

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims,
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # NN setup
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

# Create a reinforcement learning agent
agent1 = ReinforcementLearningAgent()
agent2 = RandomPlayer()
players = [agent1, agent2]

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 100
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001  

# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # State defines which player it needs input from
        curr_player = state['player']

        action = curr_player.select_action(state)
        next_state, reward, done, _ = env.step(action)

        state = next_state