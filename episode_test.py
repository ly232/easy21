"""End-to-end tests for overall episodes.

Consider this as the integraiton test. Episode class wires Agent, Game, and ControlStrategy
altogether.
"""

from episode import Episode, Action
from control_strategy import MonteCarloControlStrategy, SarsaLambdaControlStrategy
from pathlib import Path

import pandas as pd
import pickle
import tqdm


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def test_episode_playthrough_random_action() -> None:
    episode = Episode()
    episode.run()
    assert episode.is_terminal()
    print(f'random strategy trajectory: {episode.strategy.trajectory}')


def test_episode_monte_carlo_strategy() -> None:
    monte_carlo_strategy = MonteCarloControlStrategy()
    filepath = Path('MonteCarloControlStrategy.pkl')
    if filepath.exists():
        with open(filepath, 'rb') as f:
            monte_carlo_strategy = pickle.load(f)
    else:
        for _ in tqdm.tqdm(range(100000)):
            episode = Episode(strategy=monte_carlo_strategy)
            episode.run()
        monte_carlo_strategy.persist()
    # print(monte_carlo_strategy.trajectory)
    # print(monte_carlo_strategy.get_plot_df())
    monte_carlo_strategy.plot_optimal_value()

# def test_episode_sarsa_lambda_strategy() -> None:
#     sarsa_lambda_strategy = SarsaLambdaControlStrategy(lmda=0.9)
#     for _ in tqdm.tqdm(range(1000)):
#         episode = Episode(strategy=sarsa_lambda_strategy)
#         trajectory = episode.run()
