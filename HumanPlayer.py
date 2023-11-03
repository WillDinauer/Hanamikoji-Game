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

        self.card_table = {0: colored("2", "magenta"),
                           1: colored("2", "red"),
                           2: colored("2", "yellow"),
                           3: colored("3", "blue"),
                           4: colored("3", "white"),
                           5: colored("4", "green"),
                           6: colored("5", "light_cyan")}

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
        while idx not in ["1", "2", "3"]:
            print(f"Choose a card from {cards}")
            idx = input(f"Which card do you want (1, 2, 3): ")
        card = cards.pop(int(idx)-1)
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
        while idx not in ["1", "2"]:
            print(f"Groups: {groups}")
            idx = input(f"Pick a group (1 or 2): ")
        group = groups.pop(int(idx)-1)
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
        # TODO: Refactor this loop...automate some of these checks...ALSO this won't work with colored card conversion...
        while True:
            move = input("Play a move (ie. a3, b53, c534, d4353) or type 'h' to see the board: ")
            try:
                code = move[0].lower()
                if code in self.moves_left:
                    # Copy of hand for validation
                    hand = self.hand.copy()
                    if code.lower() == "h":
                        self.board.printBoard()
                        print(f"(I am player {self.side + 1})")
                    elif code == "a":
                        card = int(move[1])
                        # Try removing the card from hand...
                        hand.remove(card)
                        self.hand.remove(card)
                        self.move0(card)
                        return
                    elif code == "b":
                        card1 = int(move[1])
                        hand.remove(card1)
                        card2 = int(move[2])
                        hand.remove(card2)
                        self.hand.remove(card1)
                        self.hand.remove(card2)
                        self.move1([card1, card2])
                        return
                    elif code == "c":
                        card1 = int(move[1])
                        hand.remove(card1)
                        card2 = int(move[2])
                        hand.remove(card2)
                        card3 = int(move[3])
                        hand.remove(card3)
                        self.hand.remove(card1)
                        self.hand.remove(card2)
                        self.hand.remove(card3)
                        self.move2(opponent, [card1, card2, card3])
                        return
                    elif code == "d":
                        card1 = int(move[1])
                        hand.remove(card1)
                        card2 = int(move[2])
                        hand.remove(card2)
                        card3 = int(move[3])
                        hand.remove(card3)
                        card4 = int(move[4])
                        hand.remove(card4)
                        self.hand.remove(card1)
                        self.hand.remove(card2)
                        self.hand.remove(card3)
                        self.hand.remove(card4)
                        self.move3(opponent, [[card1, card2], [card3, card4]])
                        return
                else:
                    raise Exception("First character must be a remaining move.")
            except Exception as e:
                print(f"An exception occured: {e}")
                print("Invalid move...try again.")
                self.printMoves()

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
            print(f"{self.card_table[self.hand[i]]}, ", end="")
        print(f"{self.card_table[self.hand[-1]]}]", end="")
    
    def resetRound(self):
        self.hand = []
        self.moves_left = ["a", "b", "c", "d"]
        self.discard = None
        self.facedown = None
        random.shuffle(self.moves_left)


