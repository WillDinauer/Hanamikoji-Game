from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from utils import get_possible_actions, precompute_possibilities

# Create a reinforcement learning agent
agent1 = RandomAgent(0)
agent2 = RandomAgent(1)
players = [agent1, agent2]

storage = precompute_possibilities()

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 5
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

winner = []
# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # Environment tracks the current player
        curr_player = state['active']

        possible_actions = get_possible_actions(state, storage, curr_player)

        action = curr_player.select_action(state, possible_actions)
        next_state, reward, done = env.step(action)

        state = next_state