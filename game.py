'''Module to model the Easy21 game.'''

from dataclasses import dataclass
from enum import StrEnum
from random import randint


class Color(StrEnum):
    RED = "red"
    BLACK = "black"


@dataclass(frozen=True)
class Card:
    value: int
    color: Color


class Easy21:
    '''Class representing the Easy21 game environment.'''

    def __init__(self, red_probability: float = 1/3) -> None:
        self.draw_prob = {
            Color.RED: red_probability,
            Color.BLACK: 1 - red_probability
        }

    def draw(self, color: Color | None = None) -> Card:
        '''Draw a card from the deck.

        Args:
            color: If specified, forces the color of the drawn card, 
                otherwise color is drawn randomly.
    
        Returns:
            A Card instance drawn from the deck.
        '''
        value_rv = randint(1, 10)
        if color is None:
            color_rv = randint(1, 3)
            color = Color.RED if color_rv == 1 else Color.BLACK
            return Card(value=value_rv, color=color)
        else:
            return Card(value=value_rv, color=color)
