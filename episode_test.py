"""End-to-end tests for overall episodes.

Consider this as the integraiton test. Episode class wires Agent, Game, and ControlStrategy
altogether.
"""

from episode import Episode, Action
from control_strategy import RandomControlStrategy, MonteCarloControlStrategy

import pandas as pd
import tqdm


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


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
    trajectory = episode.run()
    assert episode.is_terminal()
    print(trajectory)

def test_episode_monte_carlo_strategy() -> None:
    monte_carlo_strategy = MonteCarloControlStrategy()
    for _ in tqdm.tqdm(range(10000)):
        episode = Episode(strategy=monte_carlo_strategy)
        trajectory = episode.run()
        monte_carlo_strategy.policy_iteration(trajectory=trajectory)
    print(monte_carlo_strategy.q)
