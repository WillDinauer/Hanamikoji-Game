import HanamikojiEnvironment
import ReinforcementLearningAgent
import random


# Create a Hanamikoji environment
env = HanamikojiEnvironment()

# Create a reinforcement learning agent
agent = ReinforcementLearningAgent()

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

        # Epsilon-greedy
        exploration_rate_threshold = random.uniform(0, 1)
        

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state