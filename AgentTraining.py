from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent
from utils import get_possible_actions, precompute_possibilities, plot_learning, plot_single
import numpy as np

# Create a reinforcement learning agent
# agent1 = RandomAgent(0)
agent1 = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=6, n_actions=273, eps_end=0.01, input_dims=[40], lr=0.003, side=0)
agent2 = RandomAgent(1)
players = [agent1, agent2]

storage = precompute_possibilities()

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 1000000
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0001

scores, eps_history, percent = [], [], []
win, loss = 0, 0
# Train the agent
for episode in range(1, num_episodes+1):
    state = env.reset()
    done = False
    stored_state = None
    while not done:
        # Environment tracks the current player
        curr_player = state['active']

        possible_actions = get_possible_actions(state, storage, curr_player)

        action = curr_player.select_action(state, possible_actions)
        next_state, reward, done, info = env.step(action)

        if curr_player is agent1:
            if stored_state is not None:
                agent1.store_transition(stored_state, action, reward, state, done, False)
            stored_state = state
        
        # TODO: move this into env
        # Learn when the round ends
        if info['finished']:
            state = [0] * 40
            agent1.store_transition(stored_state, action, reward, state, done, True)
            actual_reward = info['p1_points']-info['p2_points']
            if done:
                if info['winner'] == agent1:
                    # Reward the agent for winning
                    actual_reward += 5
                    win += 1
                else:
                    # Punish the agent for losing
                    actual_reward -= 5
                    loss += 1
                scores.append(actual_reward)
            stored_state = None
            agent1.update_rewards(actual_reward)
            agent1.learn()

        state = next_state
    eps_history.append(agent1.epsilon)
    win_percent = win / (win + loss)
    percent.append(win_percent)
    avg_score = np.mean(scores)
    if episode % 100 == 0:
        print(f'episode: {episode} | avg_score: {avg_score} | win %: {win_percent}')
    else:
        # print(f'episode: {episode} | score: {scores[-1]}')
        pass
# Save the trained model
agent1.save_model()

# Plotting
x = [i+1 for i in range(num_episodes)]
plot_learning(x, scores, eps_history, 'TrainingPlots/DQN_Learn.png')
plot_single(x, percent, 'TrainingPlots/DQN_Win.png')