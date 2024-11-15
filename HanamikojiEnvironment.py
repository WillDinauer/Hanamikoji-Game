"""
HanamikojiEnvironment.py

Python class setting up the Hanamikoji game as an OpenAI Gym Environment. This allows for AI agents to train on the game.
The goal of the game is to win the favor of four of the seven geishas, or to reach 11 points. 
The full rulebook can be found here: https://cdn.1j1ju.com/medias/e0/90/0c-hanamikoji-rulebook.pdf

William Dinauer, November 2023
"""

import random
from Board import Board
import gym

class HanamikojiEnvironment(gym.Env):
    def __init__(self, player1, player2):
        self.board = Board()
        self.players = [player1, player2]
        # Seven geishas
        self.values = [2, 2, 2, 3, 3, 4, 5]
        # A number of cards is added to the deck equal to the number on the geisha 
        # (There are 4 cards for the geisha of value 4, 2 cards for each geisha of value 2, etc.)
        self.starting_deck = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
        self.deck = self.starting_deck.copy()
        self.current_player = None
        self.round = 0

    def reset(self):
        self.current_player = random.choice(self.players)
        self.board.resetFavor()
        self.round = 0
        self.initializeRound()
        return self.get_state()
    
    def initializeRound(self):
        self.round += 1
        
        # Reset the board 
        self.board.resetBoard()

        # Fill the deck and shuffle
        self.deck = self.starting_deck.copy()
        random.shuffle(self.deck)

        # Remove a card at random from the game
        self.removed = self.deck.pop()

        # Reset the players' info
        for player in self.players:
            player.resetRound()

        # Each player draws 6 cards
        for i in range(6):
            for player in self.players:
                player.draw(self.deck.pop())

        # Current player draws a card for the first turn
        self.current_player.draw(self.deck.pop())

    def handle_action(self, action, board):
        move = len(action)
        # Each move is only used once each round
        self.current_player.moves_left.remove(move)
        # Based on the action, update the state of the player or the board
        if move == 1:
            # Card stored facedown
            self.current_player.facedown = self.current_player.hand.pop(action[0])
        elif move == 2:
            # Cards discarded
            self.current_player.discard = self.get_buffer(action)
        elif move == 3:
            # Waiting for a response...
            board.response = True
            board.response_buffer = self.get_buffer(action)
        elif move == 4:
            # Waiting for a response...
            board.response = True
            board.response_buffer = self.get_buffer(action)

    def get_buffer(self, action):
        # Play the cards from the player's hand, making sure to remove them from the hand array
        arr = []
        for idx in action:
            arr.append(self.current_player.hand[idx])
        for num in arr:
            self.current_player.hand.remove(num)
        return arr

    def get_player_state(self, player):
        # Return the agent's internal state
        return {
            "moves_left": player.moves_left,
            "hand": player.hand,
            "facedown": player.facedown,
            "discard": player.discard,
        }
    
    def get_limited_state(self, player):
        # Return the state that your opponent is allowed to see
        return {
            "moves_left": player.moves_left,
            "hand_size": len(player.hand)
        }

    def handle_response(self, action, board, opponent_side):
        my_cards = []
        opponent_cards = []
        # Responding to move 3...select a single card of three
        if len(board.response_buffer) == 3:
            self.current_player.responses_left.remove(1)
            my_cards = [board.response_buffer.pop(action[0])]
            opponent_cards = board.response_buffer.copy()
        # Responding to move 4...select a pair of cards
        else:
            self.current_player.responses_left.remove(2)
            g1 = [board.response_buffer[0], board.response_buffer[1]]   # Group 1
            g2 = [board.response_buffer[2], board.response_buffer[3]]   # Group 2
            if action[0] == 0:
                my_cards = g1
                opponent_cards = g2
            else:
                my_cards = g2
                opponent_cards = g1

        # We are no longer waiting for a response...reset the waiting condition and buffer
        board.response_buffer = []
        board.response = False

        # Place cards on the board according to the choices
        arr = []
        for card in my_cards:
            arr.append([self.current_player.side, card])
        for card in opponent_cards:
            arr.append([opponent_side, card])
        board.place_cards(arr)

    def step(self, action):
        curr = self.current_player
        # Apply the chosen action for the current player
        if self.board.response:
            was_response = True
            self.handle_response(action, self.board, self.get_opponent().side)
        else:
            was_response = False
            self.handle_action(action, self.board)

        # Check for game termination and calculate the reward
        if self.is_round_over():
            # Reveal the hidden cards and add them to the board
            arr = []
            for player in self.players:
                arr.append([player.side, player.facedown])
            self.board.place_cards(arr)

            # Check for a winner
            winner, p1_points, p2_points = self.check_winner()
            if winner is None:
                self.initializeRound()
                return self.get_state(), self.calculate_reward(curr, winner), False, {"finished": True, "p1_points": p1_points, "p2_points": p2_points}
            else:
                return self.get_state(), self.calculate_reward(curr, winner), True, {"winner": winner, "p1_points": p1_points, "p2_points": p2_points, "finished": True}
        else:
            # If we're not waiting for a response, swap the current player and have them draw a card
            if not was_response:
                self.current_player = self.get_opponent()
            if not self.board.response:
                self.current_player.draw(self.deck.pop())
            return self.get_state(), 0, False, {"finished": False}

    def get_state(self):
        # Return the game state representation (e.g., cards in hand, board state, current player)
        state = {
            "active": self.current_player,
            "board": self.board.get_state(),
            "current_player": self.get_player_state(self.current_player),
            "opponent": self.get_limited_state(self.get_opponent())
        }
        return state

    def is_round_over(self):
        # Check if the game is over (e.g., a player has won)
        for player in self.players:
            if not player.finished():
                return False
        return True
    
    # Check if player1 or player2 has won
    def check_winner(self):
        # Calculate points and influenced geishas based on the state of the board
        p1_points, p2_points = 0, 0
        p1_ct, p2_ct = 0, 0
        for i in range(7):
            winning = self.board.whos_winning(i)
            if winning == 1:
                p1_points += self.values[i]
                p1_ct += 1
            elif winning == -1:
                p2_points += self.values[i]
                p2_ct += 1

        # Has anyone accumulated 11 or more points?
        if p1_points >= 11:
            return self.players[0], p1_points, p2_points
        if p2_points >= 11:
            return self.players[1], p1_points, p2_points
        
        # Has anyone won the favor of 4 or more geishas?
        if p1_ct >= 4:
            return self.players[0], p1_points, p2_points
        if p2_ct >= 4:
            return self.players[1], p1_points, p2_points
        
        # There is no winner...yet
        return None, p1_points, p2_points

    def calculate_reward(self, curr, winner):
        # Define a reward function based on game outcome
        if winner is None:
            return 0  # No winner yet, the game is ongoing
        elif winner == curr:
            return 1  # Current player wins
        else:
            return -1  # Opponent wins
        
    def get_opponent(self):
        return self.players[1] if self.current_player == self.players[0] else self.players[0]