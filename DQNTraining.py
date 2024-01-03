from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from MaxAgent import MaxAgent
from MinAgent import MinAgent
from DQNAgent import DQNAgent
from DiverseAgent import DiverseAgent
from utils import get_possible_actions, precompute_possibilities, plot_learning, plot_single
import numpy as np
import csv

# Create a reinforcement learning agent
# agent1 = RandomAgent(0)
agent1 = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=12, n_actions=273, eps_end=0.01, input_dims=[40], lr=0.005, side=0, eps_dec=5e-5)
agent2 = RandomAgent(1)
players = [agent1, agent2]

storage = precompute_possibilities()

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 100000
learning_rate = 0.2
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
                    actual_reward += 0
                    win += 1
                else:
                    # Punish the agent for losing
                    actual_reward -= 0
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
        for move_num in agent1.move_freq:
            total = sum(move_num)
            percs = [int(move_num[i]*100/total)/100 for i in range(4)]
            print(percs)
    else:
        # print(f'episode: {episode} | score: {scores[-1]}')
        pass
# Save the trained model
agent1.save_model('Models/DQN_Model_baseline.pth')

x = [i+1 for i in range(num_episodes)]
# Save data to a CSV file
with open('TrainingData/dqn_vs_baseline.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['Episode', 'Score', 'Win %', 'Eps History'])
    
    # Write data rows
    for episode, score, perc, eps in zip(x, scores, percent, eps_history):
        csv_writer.writerow([episode, score, perc, eps])

# Plotting
plot_learning(x, scores, eps_history, 'TrainingPlots/DQN_Learn_baseline.png')
plot_single(x, percent, 'TrainingPlots/DQN_Win_baseline.png')