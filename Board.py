

# Publicly displayed cards on the board
class Board:
    def __init__(self) -> None:
        self.player1_side = [0 for i in range(7)]
        self.player2_side = [0 for i in range(7)]
        self.favor = [0 for i in range(7)]

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
            return 1
        # Player 2 is winning this favor
        elif self.player2_side[idx] > self.player1_side[idx]:
            return -1
        # Tied influence
        return 0
    
    def printBoard(self):
        print(f"  - - - - - - - - - - - - - - - - - - - -  ")
        print(f"|  {self.player1_side[0]}  |  {self.player1_side[1]}  |  {self.player1_side[2]}  |  {self.player1_side[3]}  |  {self.player1_side[4]}  |  {self.player1_side[5]}  |  {self.player1_side[6]}  |  <- Player 1")
        print(f"| - - - - - - - - - - - - - - - - - - - - |")
        print(f"| (2) | (2) | (2) | (3) | (3) | (4) | (5) |")
        print(f"| - - - - - - - - - - - - - - - - - - - - |")
        print(f"|  {self.player2_side[0]}  |  {self.player2_side[1]}  |  {self.player2_side[2]}  |  {self.player2_side[3]}  |  {self.player2_side[4]}  |  {self.player2_side[5]}  |  {self.player2_side[6]}  |  <- Player 2")
        print(f"  - - - - - - - - - - - - - - - - - - - -  ")
        print(f"                                           ")