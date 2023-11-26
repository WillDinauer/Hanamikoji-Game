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
        print(f"possible_actions: {possible_actions} and random: {random.choice(possible_actions)}")
        return random.choice(possible_actions)

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
    
    def handle_action(self, action, board):
        move = len(action)
        self.moves_left.remove(move)
        if move == 1:
            self.facedown = self.hand.pop(action[0])
        elif move == 2:
            self.discard = self.get_buffer(action, 2)
        elif move == 3:
            board.response = True
            board.response_buffer = self.get_buffer(action, 3)
        elif move == 4:
            board.response = True
            board.response_buffer = self.get_buffer(action, 4)

    def get_buffer(self, action, n):
        arr = []
        for i in range(n):
            arr.append(self.hand.pop(action[i]-i))
        return arr

    def handle_response(self, action, board, opponent_side):
        my_cards = []
        opponent_cards = []
        if len(board.response_buffer) == 3:
            self.responses_left.remove(1)
            my_cards = [board.response_buffer.pop(action[0])]
            opponent_cards = board.response_buffer.copy()
        else:
            self.responses_left.remove(2)
            g1 = [board.response_buffer[0], board.response_buffer[1]]
            g2 = [board.response_buffer[2], board.response_buffer[3]]
            if action[0] == 0:
                my_cards = g1
                opponent_cards = g2
            else:
                my_cards = g2
                opponent_cards = g1
        board.response_buffer = []
        board.response = False

        arr = []
        for card in my_cards:
            arr.append([self.side, card])
        for card in opponent_cards:
            arr.append([opponent_side, card])
        board.place_cards(arr)

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