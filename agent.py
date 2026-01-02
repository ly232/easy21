from enum import StrEnum
from game import Color, Card, Easy21

class AgentStatus(StrEnum):
    PLAYING = "playing"
    STICKED = "sticked"
    BUSTED = "busted"

class Agent:
    def __init__(self, game: Easy21):
        self.game = game

        # At the start of the game, each player draws a black card.
        black_card = game.draw(color=Color.BLACK)
        # Black card's values are always added.
        self.value = black_card.value

        self.status = AgentStatus.PLAYING

    def stick(self) -> None:
        '''If player decides to stick, no further actions.'''
        assert self.status == AgentStatus.PLAYING
        self.status = AgentStatus.STICKED

    def hit(self) -> None:
        '''Player decides to draw another card.'''
        assert self.status == AgentStatus.PLAYING
        card: Card = self.game.draw()
        delta_value = card.value if card.color == Color.BLACK else -card.value
        self.value += delta_value
        if self.value < 1 or self.value > 21:
            self.status = AgentStatus.BUSTED
