import random
from termcolor import colored

class HumanPlayer:
    def __init__(self, board, name, side) -> None:
        self.hand = []
        self.moves_left = ["a","b","c","d"]
        self.board = board
        self.name = name
        self.facedown = None
        self.discard = None
        self.side = side

    # Save one card facedown
    def move0(self, card):
        self.facedown = card
        self.moves_left.remove("a")
        return
    
    # Discard two cards from the game
    def move1(self, cards):
        self.discard = cards
        self.moves_left.remove("b")
        return
    
    # Choose 3 cards and play them face-up. The opponent chooses one to play for themself, and you play the other two
    def move2(self, opponent, cards):
        cards_left = opponent.move2Response(cards)
        publish = []
        for card in cards_left:
            publish.append([self.side, card])
        self.board.placeCards(publish)
        self.moves_left.remove("c")
        return


    # Respond to a request from the opponent
    def move2Response(self, cards):
        print("Opponent picked move 'c'")
        idx = -1
        while idx not in ["0", "1", "2"]:
            print(f"Choose a card from [", end="")
            for i in range(len(cards)-1):
                print(f"{self.board.card_table[cards[i]]}, ", end="")
            print(f"{self.board.card_table[cards[-1]]}]")
            idx = input(f"Which card do you want (0, 1, 2): ")
        card = cards.pop(int(idx))
        self.board.placeCards([[self.side, card]])
        return cards
                            

    # Choose 4 cards and split them into 2 piles of 2. Your opponent takes one pile, and you receive the other.
    def move3(self, opponent, groups):
        group_left = opponent.move3Response(groups)
        publish = []
        for card in group_left:
            publish.append([self.side, card])
        self.moves_left.remove("d")
        self.board.placeCards(publish)

    # Make a selection from the two groups, return the unselected group of cards
    def move3Response(self, groups):
        print("Opponent picked move 'd'")
        idx = -1
        while idx not in ["0", "1"]:
            print(f"Groups: ", end="")
            for group in groups:
                print(f"[{self.board.card_table[group[0]]}, {self.board.card_table[group[1]]}] ", end="")
            idx = input(f"\nPick a group (0 or 1): ")
        group = groups.pop(int(idx))
        publish = []
        for card in group:
            publish.append([self.side, card])
        self.board.placeCards(publish)
        return groups.pop()

    def draw(self, card):
        self.hand.append(card)
        self.hand.sort()

    # Pick a move and play it
    def playMove(self, opponent):
        move = -1
        self.printMoves()

        while move != 0:
            move = input("Play a move in the format [move][index] (i.e. a0, b03, c124, d0356) or 'h' to see the board: ")
            move = self.handleMove(move, opponent)
            
    # Handle the move passed by the human player
    def handleMove(self, move, opponent):
        try:
            code = move[0].lower()
            if code in self.moves_left:
                if code.lower() == "h":
                    self.board.printBoard()
                    print(f"(I am player {self.side + 1})")
                elif code == "a":
                    cards = self.extractCards(move, 1)
                    self.move0(cards)
                    return 0
                elif code == "b":
                    cards = self.extractCards(move, 2)
                    self.move1(cards)
                    return 0
                elif code == "c":
                    cards = self.extractCards(move, 3)
                    self.move2(opponent, cards)
                    return 0
                elif code == "d":
                    cards = self.extractCards(move, 4)
                    self.move3(opponent, [[cards[0], cards[1]], [cards[2], cards[3]]])
                    return 0
            else:
                raise Exception("First character must be a remaining move.")
        except Exception as e:
            print(f"An exception occured: {e}")
            self.printMoves()
            return 1

    # Given a move, return the cards as an array and remove them from the hand
    def extractCards(self, move, length):
        if len(move)-1 != length:
            raise Exception("Invalid move format")
        indices = []
        for i in range(1, length+1):
            indices.append(int(move[i]))
    
        if self.validateIndices(indices):
            cards = []
            for idx in indices:
                cards.append(self.hand[idx])
            for card in cards:
                self.hand.remove(card)
            return cards
        raise Exception("Invalid indices")

    # Validate the indices given by the player...each index must be a position in the hand.
    # Also ensure there are no duplicate indices given.
    def validateIndices(self, indices):
        seen = []
        for idx in indices:
            if idx >= len(self.hand) or idx in seen:
                return False
            seen.append(idx)
        return True
    
    # Helper function with colored move descriptions
    def printMoves(self):
        avail = 'green'
        used = 'red'
        print(colored("a: Pick a card from your hand to play face down. It will score this round.", avail if "a" in self.moves_left else used))
        print(colored("b: Pick two cards from your hand to REMOVE from play facedown. They will not score this round.", avail if "b" in self.moves_left else used))
        print(colored("c: Pick three cards from your hand to play face up. Your opponent picks one, and you get the remaining two.", avail if "c" in self.moves_left else used))
        print(colored("d: Pick four cards from your hand to play faceup in two groups of two. Your opponent picks one group, and you receive the remaining two cards.", avail if "d" in self.moves_left else used))
        print(f"Player hand: ", end="")
        self.printHand()
        print(f" | You are Player{self.side+1}")

    def printHand(self):
        print("[", end="")
        for i in range(len(self.hand)-1):
            print(f"{self.board.card_table[self.hand[i]]}, ", end="")
        print(f"{self.board.card_table[self.hand[-1]]}]", end="")
    
    def resetRound(self):
        self.hand = []
        self.moves_left = ["a", "b", "c", "d"]
        self.discard = None
        self.facedown = None
        random.shuffle(self.moves_left)


