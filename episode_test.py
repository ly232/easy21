from episode import Episode, Action

import random

def test_episode_playthrough_always_stick() -> None:
    episode = Episode()
    episode.step(Action.STICK)
    assert episode.is_terminal()
    print(episode.trajectory)

def test_episode_playthrough_always_hit() -> None:
    episode = Episode()
    while not episode.is_terminal():
        episode.step(Action.HIT)
    assert episode.is_terminal()
    print(episode.trajectory)


def test_episode_playthrough_random_action() -> None:
    episode = Episode()
    while not episode.is_terminal():
        action = random.choice(list(Action))
        episode.step(action)
    assert episode.is_terminal()
    print(episode.trajectory)