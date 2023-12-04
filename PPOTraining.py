from HanamikojiEnvironment import HanamikojiEnvironment
from RandomAgent import RandomAgent
from PPOAgent import PPOAgent
from utils import get_possible_actions, precompute_possibilities, plot_learning, plot_single
import numpy as np

if __name__ == '__main__':
    N = 6
    batch_size = 1
    n_epochs = 6
    alpha = 0.0003
    gamma = 0.99
    gae_lambda = 0.95
    policy_clip = 0.1

    # Create a reinforcement learning agent
    agent1 = PPOAgent(side=0, gamma=gamma, gae_lambda=gae_lambda, policy_clip=policy_clip, n_actions=273, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=[40])
    agent2 = RandomAgent(1)
    players = [agent1, agent2]

    storage = precompute_possibilities()

    # Create a Hanamikoji environment
    env = HanamikojiEnvironment(agent1, agent2)

    # Training parameters
    num_episodes = 100

    scores, percent = [], []
    win, loss = 0, 0

    n_steps = 0
    learn_iters = 0
    # Train the agent
    for episode in range(1, num_episodes+1):
        state = env.reset()
        done = False
        print(f"ep: {episode}")

        while not done:
            # Environment tracks the current player
            curr_player = state['active']

            possible_actions = get_possible_actions(state, storage, curr_player)

            if curr_player is agent1:
                action, prob, val = curr_player.choose_action(state, possible_actions)
                next_state, reward, done, info = env.step(action)
                agent1.remember(state, action, prob, val, reward, done)
            else:
                action = curr_player.select_action(state, possible_actions)
                next_state, reward, done, info = env.step(action)
            
            if info['finished']:
                state = [0] * 40
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
                agent1.update_rewards(actual_reward, 6)
                agent1.learn()

            state = next_state
        win_percent = win / (win + loss)
        percent.append(win_percent)
        avg_score = np.mean(scores)
        if episode % 10 == 0:
            print(f'episode: {episode} | avg_score: {avg_score} | win %: {win_percent}')
        else:
            # print(f'episode: {episode} | score: {scores[-1]}')
            pass
    # Save the trained model
    agent1.save_models()

    # Plotting
    x = [i+1 for i in range(num_episodes)]
    plot_single(x, scores, 'TrainingPlots/PPO_Scores.png')
    plot_single(x, percent, 'TrainingPlots/PPO_Win.png')