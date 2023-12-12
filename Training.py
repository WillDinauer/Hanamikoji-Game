from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from DiverseAgent import DiverseAgent
from MaxAgent import MaxAgent
from MinAgent import MinAgent
from CleverAgent import CleverAgent
from utils import get_possible_actions, precompute_possibilities, plot_learning, plot_single
import numpy as np
import csv

# Create a reinforcement learning agent
# agent1 = RandomAgent(0)
agent1 = MaxAgent(0)
agent2 = MinAgent(1)
players = [agent1, agent2]

storage = precompute_possibilities()

# Create a Hanamikoji environment
env = HanamikojiEnvironment(agent1, agent2)

# Training parameters
num_episodes = 100000

percent, scores = [], []
win, loss = 0, 0
# Train the agent
for episode in range(1, num_episodes+1):
    state = env.reset()
    done = False
    while not done:
        # Environment tracks the current player
        curr_player = state['active']

        possible_actions = get_possible_actions(state, storage, curr_player)

        action = curr_player.select_action(state, possible_actions)
        # print(f"action: {action}")
        next_state, reward, done, info = env.step(action)
        # print(f"buffer: {env.board.response_buffer}")
        # env.board.printBoard()

        if info['finished']:
            state = [0] * 40
            actual_reward = info['p1_points']-info['p2_points']
            if done:
                if info['winner'] == agent1:
                    actual_reward += 5
                    win += 1
                else:
                    actual_reward -= 5
                    loss += 1
                scores.append(actual_reward)

        state = next_state
    win_percent = win / (win + loss)
    percent.append(win_percent)
    avg_score = np.mean(scores)
    if episode % 100 == 0:
        print(f'episode: {episode} | avg_score: {avg_score} | win %: {win_percent}')
    else:
        # print(f'episode: {episode} | score: {scores[-1]}')
        pass

x = [i+1 for i in range(num_episodes)]

# Save data to a CSV file
with open('TrainingData/min_vs_random.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['Episode', 'Score', 'Win %'])
    
    # Write data rows
    for episode, score, eps in zip(x, scores, percent):
        csv_writer.writerow([episode, score, eps])

# Plotting
plot_single(x, scores, 'TrainingPlots/Diverse_Learn.png')
plot_single(x, percent, 'TrainingPlots/Diverse_Win.png')