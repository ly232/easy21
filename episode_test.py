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

import matplotlib.pyplot as plt
import pytest


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#
# Test random control strategy.
#
def test_episode_playthrough_random_action() -> None:
    episode = Episode()
    episode.run()
    assert episode.is_terminal()
    print(f'random strategy trajectory: {episode.strategy.trajectory}')


#
# Test Monte Carlo control strategy.
#
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
    monte_carlo_strategy.plot_optimal_value(show=False)

#
# Test SARSA(λ) control strategy.
#
SARSA_LAMBDAS = [0.1 * i for i in range(11)]
@pytest.fixture(scope="module")
def sarsa_figure():
    # Create a vertical column of subplots: one subplot per lambda value.
    fig, axes = plt.subplots(len(SARSA_LAMBDAS), 1, figsize=(6, 4 * len(SARSA_LAMBDAS)))
    yield fig, axes
    fig.tight_layout()
    fig.savefig("SarsaLambdaControlStrategy.png")

@pytest.mark.parametrize("lmda", SARSA_LAMBDAS)
def test_episode_sarsa_lambda_strategy(lmda, sarsa_figure) -> None:
    _, axes = sarsa_figure
    idx = SARSA_LAMBDAS.index(lmda)
    ax = axes[idx]
    ax.set_title(f'SARSA(λ) Optimal Value Function (λ={lmda})')

    sarsa_lambda_strategy = SarsaLambdaControlStrategy(lmda=lmda)
    filepath = Path('SarsaLambdaControlStrategy.pkl')
    if filepath.exists():
        with open(filepath, 'rb') as f:
            sarsa_lambda_strategy = pickle.load(f)
    else:
        for _ in tqdm.tqdm(range(100000)):
            episode = Episode(strategy=sarsa_lambda_strategy)
            episode.run()
        sarsa_lambda_strategy.persist()
    # print(sarsa_lambda_strategy.trajectory)
    # print(sarsa_lambda_strategy.get_plot_df())
    sarsa_lambda_strategy.plot_optimal_value(ax=ax, show=False)
