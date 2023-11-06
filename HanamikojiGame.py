import random
from Board import Board
from RandomPlayer import RandomPlayer
from HumanPlayer import HumanPlayer

class HanamikojiGame:

    # Initialize the parameters necessary for the game
    def __init__(self, player1=None, player2=None, board=None) -> None:
        self.values = [2, 2, 2, 3, 3, 4, 5]
        self.p1 = player1
        self.p2 = player2
        self.starting_deck = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
        self.deck = self.starting_deck.copy()
        self.starting_moves = [0, 1, 2, 3]
        self.board = board
        self.round = 0
    
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
        self.p1.resetRound()
        self.p2.resetRound()

        # Each player draws 6 cards
        for i in range(6):
            self.p1.draw(self.deck.pop())
            self.p2.draw(self.deck.pop())
    
    def playRound(self, first_player, second_player):
        print(f"First player: {first_player.name} | Second player: {second_player.name}")
        # Each player has 4 moves
        for i in range(4):
            # Player 1 draws and plays a remaining move
            first_player.draw(self.deck.pop())
            first_player.playMove(second_player)
            
            print(f"move: {2*i+1}")
            self.board.printBoard()

            # Player 2 draws and plays a remaining move
            second_player.draw(self.deck.pop())
            second_player.playMove(first_player)
            print(f"move: {2*i+2}")
            self.board.printBoard()

        print(f"Facedown cards: {first_player.name} - {first_player.facedown}, {second_player.name} - {second_player.facedown}")
        self.board.placeCards([[first_player.side, first_player.facedown], [second_player.side, second_player.facedown]])
        print(f"After reveals:")
        self.board.printBoard()
        # Has anyone won yet?
        return self.checkWinner()
    
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


    def handleGame(self):  
        if random.randint(0, 1) == 0:
            first = self.p1
            second = self.p2
        else:
            first = self.p2
            second = self.p1

        # Keep playing rounds until there is a winner
        winner = None
        while winner is None:
            self.initializeRound()
            winner = self.playRound(first, second)

            if winner != None:
                print(f"The winner is {winner.name} after {self.round} rounds!")
                return winner
            
            if first == self.p1:
                first = self.p2
                second = self.p1
            else:
                first = self.p1
                second = self.p2


if __name__ == "__main__":
    board = Board()
    player1 = RandomPlayer(board, "p1", 0)
    player2 = HumanPlayer(board, "p2", 1)
    hkg = HanamikojiGame(player1, player2, board)
    hkg.handleGame()