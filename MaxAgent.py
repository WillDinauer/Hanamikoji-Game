import random

class MaxAgent:
    def __init__(self, side):
        # Hanamikoji parameters
        self.hand = []
        self.moves_left = [1, 2, 3, 4]
        self.responses_left = [1, 2]
        self.facedown = None
        self.discard = None
        self.side = side

    def select_action(self, observation, possible_actions):
        # Choose an available action randomly, as long as it is diverse (if possible)
        resp = observation['board']['response_buffer']
        if len(resp) == 3:
            best_idx = -1
            maximum = 0
            for i in range(3):
                if resp[i] > maximum:
                    best_idx = i
                    maximum = resp[i]
            return possible_actions[best_idx]

        if len(resp) == 4:
            if resp[0] + resp[1] > resp[2] + resp[3]:
                return possible_actions[0]
            else:
                return possible_actions[1]
        

        action_len = random.choice(self.moves_left)
        maximum = -1
        best_action = 0
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