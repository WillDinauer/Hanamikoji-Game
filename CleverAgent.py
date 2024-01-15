import random

class CleverAgent:
    def __init__(self, side):
        # Hanamikoji parameters
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.facedown = None
        self.discard = None
        self.side = side
        self.grid = [[0.09, 0.54, 0.24, 0.11],
                    [0.35, 0.17, 0.3, 0.17],
                    [0.38, 0.14, 0.24, 0.22],
                    [0.17, 0.13, 0.2, 0.48]]

    def select_action(self, observation, possible_actions):
        # Choose an available action randomly, as long as it is diverse (if possible)
        resp = observation['board']['response_buffer']

        # For a response to move 3, select the maximum card
        if len(resp) == 3:
            best_idx = -1
            maximum = 0
            for i in range(3):
                if resp[i] > maximum:
                    best_idx = i
                    maximum = resp[i]
            return possible_actions[best_idx]

        # For a response to move 4, select the pile with the greater sum
        if len(resp) == 4:
            if resp[0] + resp[1] > resp[2] + resp[3]:
                return possible_actions[0]
            else:
                return possible_actions[1]
        
        # Create an array of weights for the remaining possible actions left
        w = []
        for i in range(1, 5):
            if i in self.moves_left:
                w.append(self.grid[4-len(self.moves_left)][i-1])

        # Select an action length from these learned weights
        action_len = random.choices(self.moves_left, weights=tuple(w), k=1)[0]
        maximum = -1
        best_action = 0
        # Select the maximum sum of the action with the selected length
        for action in possible_actions:
            if len(action) == action_len:
                total = 0
                for idx in action:
                    total += self.hand[idx]
                if total > maximum:
                    maximum = total
                    best_action = action
        return best_action

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