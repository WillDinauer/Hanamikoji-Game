from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent
from utils import get_possible_actions, precompute_possibilities

# Create a reinforcement learning agent
# agent1 = RandomAgent(0)
agent1 = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=6, n_actions=273, eps_end=0.01, input_dims=[40], lr=0.003, side=0)
agent2 = RandomAgent(1)
players = [agent1, agent2]

storage = precompute_possibilities()

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 1
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

scores, eps_history = [], []

# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        # Environment tracks the current player
        curr_player = state['active']

        possible_actions = get_possible_actions(state, storage, curr_player)

        action = curr_player.select_action(state, possible_actions)
        next_state, reward, done, info = env.step(action)

        # TODO: move this into env
        # Learn when the round ends
        if False:
            actual_reward = info['p1_points']-info['p2_points']
            if done:
                if info['winner'] == agent1:
                    # Reward the agent for winning
                    reward += 5
                else:
                    # Punish the agent for losing
                    reward -= 5
            agent1.update_rewards(actual_reward)


        state = next_state

    scores.append(score)