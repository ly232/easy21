from enum import StrEnum
from game import Color, Card, Easy21


class ParticipantStatus(StrEnum):
    PLAYING = "playing"
    STICKED = "sticked"
    BUSTED = "busted"


class Participant:
    """Models a participant (player or dealer) in the Easy21 game.

    Note this is intentionally not named 'Agent'. Agent is reserved for RL environment
    interactions. See e.g. dm_env_adapter.py for an example Agent implementation
    against Depedmind's environment API.
    """

    def __init__(self, game: Easy21):
        self.game = game

        # At the start of the game, each player draws a black card.
        black_card = game.draw(color=Color.BLACK)
        # Black card's values are always added.
        self.value = black_card.value

        self.status = ParticipantStatus.PLAYING

    def stick(self) -> None:
        """If player decides to stick, no further actions."""
        assert self.status == ParticipantStatus.PLAYING
        self.status = ParticipantStatus.STICKED

    def hit(self) -> None:
        """Player decides to draw another card."""
        assert self.status == ParticipantStatus.PLAYING
        card: Card = self.game.draw()
        delta_value = card.value if card.color == Color.BLACK else -card.value
        self.value += delta_value
        if self.value < 1 or self.value > 21:
            self.status = ParticipantStatus.BUSTED
