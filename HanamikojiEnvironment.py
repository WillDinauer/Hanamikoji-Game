import random
from Board import Board
import gym
from gym import logger, spaces
import numpy as np

class HanamikojiEnvironment(gym.Env):
    def __init__(self, player1, player2):
        self.board = Board()
        self.players = [player1, player2]
        self.starting_deck = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
        self.deck = self.starting_deck.copy()
        self.current_player = None
        self.round = 0
        self.storage = self.precompute_possibilities()

    def get_possible_actions(self, state):
        # Determine the available actions based on the current state

        # Response action
        if state['opponent']['offer'] > 0:
            # Possible actions are the indices of choice
            if len(state['opponent']['selection'] == 4):
                return [1, 2]
            else:
                return [1, 2, 3]

        # Standard action    
        possible_actions = self.storage[len(self.current_player.hand), tuple(self.current_player.moves_left)]
    
        return possible_actions

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


    def step(self, action):
        # Apply the chosen action for the current player
        self.current_player.perform_action(action, self.board, self.players)

        # Check for game termination and calculate the reward
        if self.is_game_over():
            winner = self.get_winner()
            return self.get_state(), self.calculate_reward(self.current_player, winner), True
        else:
            self.current_player = self.get_opponent()
            return self.get_state(), 0, False

    def get_state(self):
        # Return the game state representation (e.g., cards in hand, board state, current player)
        state = {
            "board": self.board.get_state(),
            "current_player": self.current_player.get_state(),
            "opponent": self.get_opponent().get_limited_state()
        }
        return state

    def is_game_over(self):
        # Check if the game is over (e.g., a player has won)
        for player in self.players:
            if not player.finished():
                return False
        return True

    def get_winner(self):
        # Determine the winner
        p1_score, p2_score = self.board.calculate_scores()
        if p1_score >= 11:
            return self.players[0]
        elif p2_score >= 11:
            return self.players[1]
        return None
    
    # Check if player1 or player2 has won
    def checkWinner(self):
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
            return self.p1
        if p2_points >= 11:
            return self.p2
        
        print(f"Favor: {self.board.favor}")
        if p1_ct >= 4:
            return self.p1
        if p2_ct >= 4:
            return self.p2
        
        # There is no winner...yet
        return None

    def calculate_reward(self, winner):
        # Define a reward function based on game outcome
        if winner is None:
            return 0  # No winner yet, the game is ongoing
        elif winner == self.players[0]:
            return 1  # Player 1 (AI) wins
        else:
            return -1  # Player 2 (opponent) wins

    def get_opponent(self):
        return self.players[1] if self.current_player == self.players[0] else self.players[0]
    
    def precompute_possibilities(self):
        storage = {}
        possible_ml = []
        ml = []

        def recursive_ml(idx):
            if idx == 4:
                if len(ml) > 0:
                    possible_ml.append(ml.copy())
                return
            recursive_ml(idx+1)
            ml.append(idx)
            recursive_ml(idx+1)
            ml.remove(idx)

        recursive_ml(0)

        for i in range(1, 7):
            for ml in possible_ml:
                actions = []
                for move in ml:
                    if move == 0:
                        for j in range(i):
                            actions.append(j)
                    if move == 1:
                        for j in range(i):
                            for k in range(j+1, i):
                                actions.append((j, k))
                    if move == 2:
                        for j in range(i):
                            for k in range(j+1, i):
                                for l in range(k+1, i):
                                    actions.append((j, k, l))
                    if move == 3:
                        for j in range(i):
                            for k in range(j+1, i):
                                for l in range(k+1, i):
                                    for m in range(l+1, i):
                                        actions.append((j, k, l, m))
                                        actions.append((j, l, k, m))
                                        actions.append((j, m, k, l))
                storage[i, tuple(ml)] = actions
        return storage