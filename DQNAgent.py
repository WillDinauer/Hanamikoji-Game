"""
DQNAgent.py

Python class for a DQNAgent that plays Hanamikoji. The agent chooses actions using Deep Q-Learning.
The DQN is a deep neural network with 5 linear layers: an input layer, 3 hidden layers, and an output layer.

William Dinauer, November 2023
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import create_action_dict

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions

        # NN setup
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions

class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, side,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.action_dict, self.int_dict = create_action_dict()

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=1024, fc2_dims=1024, fc3_dims=512, fc4_dims=512)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

        # Hanamikoji parameters
        self.side = side
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.facedown = -1
        self.discard = [-1, -1]

        self.move_freq = [[0 for i in range(4)] for j in range(4)]

    def convert_state(self, observation):
        # Convert the observation into a normalized feature vector
        hand = observation['current_player']['hand']
        moves = observation['current_player']['moves_left']
        facedown = observation['current_player']['facedown']
        discard = observation['current_player']['discard']
        opponent_moves = observation['opponent']['moves_left']
        max_hand_size = 7
        max_moves_size = 4

        hand = hand[:max_hand_size] + [0] * (max_hand_size - len(hand))
        moves = moves[:max_moves_size] + [0] * (max_moves_size - len(moves))
        opponent_moves = opponent_moves[:max_moves_size] + [0] * (max_moves_size - len(opponent_moves))

        # 7 + 4 + 1 + 2 + 7 + 7 + 7 + 4 + 1 = 40 parameters
        observation_values = hand + moves + [facedown] + discard \
            + observation['board']['player1_side'] + observation['board']['player2_side'] + observation['board']['favor'] \
            + opponent_moves + [observation['opponent']['hand_size']]
        return observation_values

    def select_action(self, observation, possible_actions):
        # Epsilon-greedy exploit-explore
        if np.random.random() > self.epsilon:
            observation_values = self.convert_state(observation)

            state = T.tensor([observation_values], dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            
            # Convert the possible actions to their associated number
            possible_actions = [self.convert_action(a) for a in possible_actions]
            
            # Filter Q-values for allowed actions
            # allowed_q_values = actions[:, possible_actions]
            allowed_q_values = T.tensor([actions[0][a] for a in possible_actions], dtype=T.float32).to(self.Q_eval.device)

            # Choose the action with the highest Q-value from the allowed actions
            action = possible_actions[T.argmax(allowed_q_values).item()]
            action = self.int_dict[action]
        else:
            # Randomly select from allowed actions
            action = random.choice(possible_actions)

        # counter to keep track of the frequency of move choices.
        if len(observation['board']['response_buffer']) == 0:
            self.move_freq[4-len(self.moves_left)][len(action)-1] += 1
        return action

    def store_transition(self, state, action, reward, next_state, done, terminal):
        state = self.convert_state(state)
        if not terminal:
            next_state = self.convert_state(next_state)
        action = str(self.convert_action(action))
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def convert_action(self, action):
        # Convert the action into acceptable input for the DQN
        if type(action) == list:
            if len(action) == 1:
                action = action[0]
            else:
                action = tuple(action)

        return self.action_dict[action]

    def update_rewards(self, reward):
        # This is a common AI problem: assignment of reward to past actions
        # Here, the reward is evenly split among all past actions for the batch size
        for i in range(self.batch_size):
            index = (self.mem_cntr-i-1) % self.mem_size
            self.reward_memory[index] = reward
    
    def save_model(self, name):
        # Save the model
        T.save(self.Q_eval, name)

    def load_model(self, filepath):
        self.Q_eval = T.load(filepath)
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def finished(self):
        # Have I played all my moves?
        if len(self.moves_left) == 0 and len(self.responses_left) == 0:
            return True
        return False
    
    def draw(self, card):
        self.hand.append(card)
        self.hand.sort()
    
    def resetRound(self):
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.discard = [-1, -1]
        self.facedown = -1