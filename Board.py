from termcolor import colored

# Publicly displayed cards on the board
class Board:
    def __init__(self) -> None:
        self.player1_side = [0 for i in range(7)]
        self.player2_side = [0 for i in range(7)]
        self.favor = [0 for i in range(7)]
        self.card_table = {0: colored("2", "magenta"),
                           1: colored("2", "red"),
                           2: colored("2", "yellow"),
                           3: colored("3", "blue"),
                           4: colored("3", "white"),
                           5: colored("4", "green"),
                           6: colored("5", "light_cyan")}
        self.response = False
        self.response_buffer = []

    # Place cards on the board.
    # arr is an array containing [player, card] pairs
    def placeCards(self, arr):
        for pair in arr:
            if pair[0] == 0:
                self.player1_side[pair[1]] += 1
            else:
                self.player2_side[pair[1]] += 1
    
    def resetBoard(self):
        self.player1_side = [0 for i in range(7)]
        self.player2_side = [0 for i in range(7)]

    def resetFavor(self):
        self.favor = [0 for i in range(7)]

    def fullReset(self):
        self.resetBoard()
        self.resetFavor()

    def whosWinning(self, idx):
        # Player 1 is winning this favor
        if self.player1_side[idx] > self.player2_side[idx]:
            self.favor[idx] = 1
            return 1
        # Player 2 is winning this favor
        elif self.player2_side[idx] > self.player1_side[idx]:
            self.favor[idx] = -1
            return -1
        # Tied influence
        return 0
    
    def printBoard(self):
        print(f"  - - - - - - - - - - - - - - - - - - - -  ")
        print(f"|  {self.player1_side[0]}  |  {self.player1_side[1]}  |  {self.player1_side[2]}  |  {self.player1_side[3]}  |  {self.player1_side[4]}  |  {self.player1_side[5]}  |  {self.player1_side[6]}  |  <- Player 1")
        print(f"| - - - - - - - - - - - - - - - - - - - - |")
        print(f"|  {'+' if self.favor[0] == 1 else ' '}  |  {'+' if self.favor[1] == 1 else ' '}  |  {'+' if self.favor[2] == 1 else ' '}  |  {'+' if self.favor[3] == 1 else ' '}  |  {'+' if self.favor[4] == 1 else ' '}  |  {'+' if self.favor[5] == 1 else ' '}  |  {'+' if self.favor[6] == 1 else ' '}  |")
        print(f"| ({self.card_table[0]}) | ({self.card_table[1]}) | ({self.card_table[2]}) | ({self.card_table[3]}) | ({self.card_table[4]}) | ({self.card_table[5]}) | ({self.card_table[6]}) |")
        print(f"|  {'+' if self.favor[0] == -1 else ' '}  |  {'+' if self.favor[1] == -1 else ' '}  |  {'+' if self.favor[2] == -1 else ' '}  |  {'+' if self.favor[3] == -1 else ' '}  |  {'+' if self.favor[4] == -1 else ' '}  |  {'+' if self.favor[5] == -1 else ' '}  |  {'+' if self.favor[6] == -1 else ' '}  |")
        print(f"| - - - - - - - - - - - - - - - - - - - - |")
        print(f"|  {self.player2_side[0]}  |  {self.player2_side[1]}  |  {self.player2_side[2]}  |  {self.player2_side[3]}  |  {self.player2_side[4]}  |  {self.player2_side[5]}  |  {self.player2_side[6]}  |  <- Player 2")
        print(f"  - - - - - - - - - - - - - - - - - - - -  ")
        print(f"                                           ")