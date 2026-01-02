from collections import Counter
from game import Color, Card, Easy21

import pytest


@pytest.mark.parametrize("color", [Color.RED, Color.BLACK])
def test_draw_card_fixed_color(color: Color) -> None:
    game = Easy21()
    for _ in range(1000):
        card: Card = game.draw(color=color)
        assert card.color == color
        assert 1 <= card.value <= 10

def test_draw_card_random_color() -> None:
    game = Easy21()
    counter = Counter()
    for _ in range(100000):
        card: Card = game.draw()
        counter[card.color] += 1
        assert 1 <= card.value <= 10
    assert game.draw_prob[Color.RED] - 1e-2 \
            < counter[Color.RED] / sum(counter.values()) \
            < game.draw_prob[Color.RED] + 1e-2
