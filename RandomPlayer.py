import random

class RandomPlayer:
    def __init__(self, board, name, side) -> None:
        self.hand = []
        self.moves_left = [0, 1, 2, 3]
        random.shuffle(self.moves_left)
        self.board = board
        self.name = name
        self.facedown = None
        self.discard = None
        self.side = side

    # Random card from hand.
    def rfh(self):
        return self.hand.pop(random.randrange(len(self.hand)))

    # Save one card facedown
    def move0(self):
        print(f"Player{self.side+1}: Playing a")
        card = self.rfh()
        self.facedown = card
        return
    
    # Discard two cards from the game
    def move1(self):
        print(f"Player{self.side+1}: Playing b")
        self.discard = [self.rfh(), self.rfh()]
        return
    
    # Choose 3 cards and play them face-up. The opponent chooses one to play for themself, and you play the other two
    def move2(self, opponent):
        print(f"Player{self.side+1}: Playing c")
        cards_left = opponent.move2Response([self.rfh(), self.rfh(), self.rfh()])
        publish = []
        for card in cards_left:
            publish.append([self.side, card])
        self.board.placeCards(publish)


    # Respond to a request from the opponent
    def move2Response(self, cards):
        card = cards.pop(random.randrange(len(cards)))
        self.board.placeCards([[self.side, card]])
        return cards
                            

    # Choose 4 cards and split them into 2 piles of 2. Your opponent takes one pile, and you receive the other.
    def move3(self, opponent):
        print(f"Player{self.side+1}: Playing d")
        group_left = opponent.move3Response([[self.rfh(), self.rfh()], [self.rfh(), self.rfh()]])
        publish = []
        for card in group_left:
            publish.append([self.side, card])
        self.board.placeCards(publish)

    # Make a selection from the two groups, return the unselected group of cards
    def move3Response(self, groups):
        selected_group = groups.pop(random.randrange(len(groups)))
        publish = []
        for card in selected_group:
            publish.append([self.side, card])
        self.board.placeCards(publish)
        return groups.pop()

    def draw(self, card):
        self.hand.append(card)

    # Pick a move and play it
    def playMove(self, opponent):
        move = self.moves_left.pop()
        if move == 0:
            self.move0()
        elif move == 1:
            self.move1()
        elif move == 2:
            self.move2(opponent)
        else:
            self.move3(opponent)

    def resetRound(self):
        self.hand = []
        self.moves_left = [0, 1, 2 ,3]
        self.discard = None
        self.facedown = None
        random.shuffle(self.moves_left)
