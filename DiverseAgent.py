import random

class DiverseAgent:
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
        i = 0
        action = random.choice(possible_actions)
        while i < 100 and not self.diverse_action(observation, action):
            action = random.choice(possible_actions)
            i += 1
        return random.choice(possible_actions)
    
    def diverse_action(self, state, action):
        if len(state['board']['response_buffer']) == 3:
            return True
        
        if len(state['board']['response_buffer']) == 4:
            if action[0] == 0:
                if state['board']['response_buffer'][0] == state['board']['response_buffer'][1]:
                    return False
                return True
            else:
                if state['board']['response_buffer'][2] == state['board']['response_buffer'][3]:
                    return False
                return True
        
        cards = []
        for idx in action:
            card = self.hand[idx]
            if card in cards:
                return False
            cards.append(card)
        return True

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