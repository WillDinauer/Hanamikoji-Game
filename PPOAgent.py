
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import random
import os
from utils import create_action_dict

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.vals = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256,
                 fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent:
    def __init__(self, n_actions, gamma, alpha, gae_lambda, policy_clip, input_dims, batch_size, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

        # Hanamikoji parameters
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.facedown = -1
        self.discard = [-1, -1]

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

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

        # print(f"{len(hand)}{len(moves)}{len(facedown)}{len(discard)}{len(observation['board']['player1_side'])}{len(observation['board']['player2_side'])}{len(observation['board']['favor'])}{len(opponent_moves)}{len([observation['opponent']['hand_size']])}")
        # print(f"{type(hand)}{type(moves)}{type(facedown)}{type(discard)}{type(observation['board']['player1_side'])}{type(observation['board']['player2_side'])}{type(observation['board']['favor'])}{type(opponent_moves)}{type([observation['opponent']['hand_size']])}")
        
        # 7 + 4 + 1 + 2 + 7 + 7 + 7 + 4 + 1 = 40 parameter
        observation_values = hand + moves + [facedown] + discard \
            + observation['board']['player1_side'] + observation['board']['player2_side'] + observation['board']['favor'] \
            + opponent_moves + [observation['opponent']['hand_size']]
        return observation_values
    
    def choose_action(self, observation, possible_actions):
        observation_values = self.convert_state(observation)
        state = T.tensor([observation_values], dtype=T.float32).to(self.Q_eval.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, done_arr, batches = \
            self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                     (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actors(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_probs(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

    def get_state(self):
        # Return the agent's internal state
        return {
            "moves_left": self.moves_left,
            "hand": self.hand,
            "facedown": self.facedown,
            "discard": self.discard,
        }
    
    def get_limited_state(self):
        # Return the state that your opponent is allowed to see
        return {
            "moves_left": self.moves_left,
            "hand_size": len(self.hand)
        }
    
    def handle_action(self, action, board):
        # print(action)
        move = len(action)
        self.moves_left.remove(move)
        if move == 1:
            self.facedown = self.hand.pop(action[0])
        elif move == 2:
            self.discard = self.get_buffer(action, 2)
        elif move == 3:
            board.response = True
            board.response_buffer = self.get_buffer(action, 3)
        elif move == 4:
            board.response = True
            board.response_buffer = self.get_buffer(action, 4)

    def get_buffer(self, action, n):
        arr = []
        for i in range(n):
            arr.append(self.hand.pop(action[i]-i))
        return arr

    def handle_response(self, action, board, opponent_side):
        my_cards = []
        opponent_cards = []
        if len(board.response_buffer) == 3:
            self.responses_left.remove(1)
            my_cards = [board.response_buffer.pop(action[0])]
            opponent_cards = board.response_buffer.copy()
        else:
            self.responses_left.remove(2)
            g1 = [board.response_buffer[0], board.response_buffer[1]]
            g2 = [board.response_buffer[2], board.response_buffer[3]]
            if action[0] == 0:
                my_cards = g1
                opponent_cards = g2
            else:
                my_cards = g2
                opponent_cards = g1
        board.response_buffer = []
        board.response = False

        arr = []
        for card in my_cards:
            arr.append([self.side, card])
        for card in opponent_cards:
            arr.append([opponent_side, card])
        board.place_cards(arr)

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