import random
from Board import Board
import gym
from gym import logger, spaces
import numpy as np

class HanamikojiEnvironment(gym.Env):
    def __init__(self, player1, player2):
        self.board = Board()
        self.players = [player1, player2]
        self.values = [2, 2, 2, 3, 3, 4, 5]
        self.starting_deck = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
        self.deck = self.starting_deck.copy()
        self.current_player = None
        self.round = 0

    def reset(self):
        self.current_player = random.choice(self.players)
        self.board.resetFavor()
        self.initializeRound()
        return self.get_state()
    
    def initializeRound(self):
        self.round += 1
        print(f"Setting up for Round #{self.round}")
        
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


    def step(self, action):
        print(action)
        curr = self.current_player
        # Apply the chosen action for the current player
        if self.board.response:
            self.current_player.handle_response(action, self.board, self.get_opponent().side)
        else:
            self.current_player.handle_action(action, self.board)

        # Check for game termination and calculate the reward
        if self.is_round_over():
            winner = self.check_winner()
            if winner is None:
                self.initializeRound()
                return self.get_state(), self.calculate_reward(curr, winner), False
            else:
                return self.get_state(), self.calculate_reward(curr, winner), True
        else:
            # If we're not waiting for a response, swap the current player and have them draw a card
            if not self.board.response:
                self.current_player = self.get_opponent()
                self.current_player.draw(self.deck.pop())
            return self.get_state(), 0, False

    def get_state(self):
        # Return the game state representation (e.g., cards in hand, board state, current player)
        state = {
            "active": self.current_player,
            "board": self.board.get_state(),
            "current_player": self.current_player.get_state(),
            "opponent": self.get_opponent().get_limited_state()
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
        p1_points, p2_points = 0, 0
        p1_ct, p2_ct = 0, 0
        for i in range(7):
            winning = self.board.whosWinning(i)
            if winning == 1:
                p1_points += self.values[i]
                p1_ct += 1
            elif winning == -1:
                p2_points += self.values[i]
                p2_ct += 1

        print(f"Points: p1 - {p1_points}, p2 - {p2_points}")
        if p1_points >= 11:
            return self.players[0]
        if p2_points >= 11:
            return self.players[1]
        
        print(f"Favor: {self.board.favor}")
        if p1_ct >= 4:
            return self.players[0]
        if p2_ct >= 4:
            return self.players[1]
        
        # There is no winner...yet
        return None

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