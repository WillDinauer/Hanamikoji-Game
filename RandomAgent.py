"""
RanomAgent.py

Agent for playing Hanamikoji that takes actions completely at random.

William Dinauer, November 2023
"""

import random

class RandomAgent:
    def __init__(self, side):
        # Hanamikoji parameters
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.facedown = None
        self.discard = None
        self.side = side

    def select_action(self, observation, possible_actions):
        # Choose an available action randomly
        # print(f"possible_actions: {possible_actions} and random: {random.choice(possible_actions)}")
        return random.choice(possible_actions)

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
        self.discard = None
        self.facedown = None