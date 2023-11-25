import random

class RLAgent:
    def __init__(self):
        # Define your RL agent's model and parameters here
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.facedown = None
        self.discard = None

    def select_action(self, possible_actions):
        # Choose an action based on the current state
        return random.choice(possible_actions)

    def learn(self, state, action, reward, next_state):
        # Implement your learning algorithm here
        pass

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
    
    def finished(self):
        # Have I played all my moves?
        if len(self.moves_left) == 0:
            return True
        return False
    
    def resetRound(self):
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.discard = None
        self.facedown = None