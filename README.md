# Hanamikoji AI Training

## What is Hanamikoji? 
This project contains an environment for training and testing AI agents to play the two player card game Hanamikoji.
Hanamikoji is a strategic game involving a deck of 21 cards where players fight for the favor of seven geishas. The game
is played over a series of rounds, where players have four turns each. When a player takes one of the four
unique actions, they cannot take the same action again for the rest of the round, limiting their options.
The first player to achieve favor of four of the seven geishas, or to aquire 11 total points, wins the game.
For a full rules explanation, check out the official rulebook: https://cdn.1j1ju.com/medias/e0/90/0c-hanamikoji-rulebook.pdf

The game is interesting to study given the nature of the actions players take. Instead of taking simple actions that
directly affect your board state, players offer a choice to their opponent. For instance, one of the moves they can make 
involves selecting three cards from their hand to play face-up. Their opponent must choose one of these cards for themselves,
leaving the remaining two for their opponent. This adds a deeper element of bluffing to the game, which makes it an interesting 
problem to study for the purposes of AI and game theory.

## Project Overview
```HanamikojiEnvironment``` contains most of the environment code, with the ```Board``` object handling some public state.
Most agents use simple heuristics to play the game, the most simplicstic being ```RandomAgent```, which makes every decision completely randomly.
```MaxAgent``` always chooses the move with the greatest sum of cards, while ```MinAgent``` chooses the action with the minimum sum.
There are two deep neural network agents, a value-based approach and a policy-based approach. Deep Q-Learning is the value-based agent,
which consists of a neural network of 5 linear layers: an input layer, three hidden layers, and an action vector as the output.
The code for this agent is in ```DQNAgent```. The other deep neural network agent is the ```PPOAgent```, which uses Proximal Policy
Optimization to learn the game. The PPO Agent has an actor and a critic network which each has a single hidden layer.

## Testing
I'd recommend playing around with the parameters of the DQN and PPO Agents, including the structure of the neural networks, to see if you
can measure improved or worsened performance. Feel free to load one of the existing models in the ```Models/``` folder. You may also
choose to develop your own AI Agent, which is perfectly acceptable given the structure of the python environment.

## About
This project was developed by William Dinauer. Feel free to reach out with any inquiries or questions.
