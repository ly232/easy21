"""End-to-end tests for overall episodes.

Consider this as the integraiton test. Episode class wires Agent, Game, and ControlStrategy
altogether.
"""

from episode import Episode, Action
from control_strategy import MonteCarloControlStrategy, SarsaLambdaControlStrategy, State, Action
from pathlib import Path
from functools import cache
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

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
    """Generates Monte Carlo action value estimate after 100K episodes."""
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
SARSA_LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
@pytest.fixture(scope="module")
def sarsa_figure():
    # Create a vertical column of subplots: one subplot per lambda value.
    fig, axes = plt.subplots(len(SARSA_LAMBDAS), 1, figsize=(6, 4 * len(SARSA_LAMBDAS)))
    yield fig, axes
    fig.tight_layout()
    fig.savefig("SarsaLambdaControlStrategy.png")

@pytest.mark.parametrize("lmda", SARSA_LAMBDAS)
def test_episode_sarsa_lambda_strategy(lmda, sarsa_figure) -> None:
    """Generates Sarsa(λ) action value estimate after 100K episodes."""
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

def test_mse_sarsa_lambda_strategy() -> None:
    """Generates MSE per λ value against Sarsa(λ) after 1k episodes.
    
    Uses MC 100K-eisode result as ground truth.
    """
    num_episodes = 1000

    def load_ground_truth() -> dict[State, dict[Action, float]]:
        filepath = Path('MonteCarloControlStrategy.pkl')
        with open(filepath, 'rb') as f:
            monte_carlo_strategy = pickle.load(f)
        return monte_carlo_strategy.q
    
    def compute_mse(q1, q2) -> float:
        """Computes MSE between two Q value dictionaries."""
        q1, q2 = deepcopy(q1), deepcopy(q2)
        mse = 0.0
        count = 0
        states = set(q1.keys()).union(set(q2.keys()))
        for state in states:
            if state.is_terminal:
                continue
            if state not in q1:
                q1[state] = {Action.HIT: 0.0, Action.STICK: 0.0}
            if state not in q2:
                q2[state] = {Action.HIT: 0.0, Action.STICK: 0.0}
            count += 2
            mse += (q1[state][Action.HIT] - q2[state][Action.HIT]) ** 2
            mse += (q1[state][Action.STICK] - q2[state][Action.STICK]) ** 2
        assert count > 0
        return mse / count

    def single_run(lmda: float) -> list[dict[State, dict[Action, float]]]:
        """Runs a single 1K-episode SARSA(λ) strategy and returns the Q values.
        
        Returns a sequence of MSE errors indexed by episode number.
        """
        sarsa_lambda_strategy = SarsaLambdaControlStrategy(lmda=lmda)
        mses = []
        ground_truth_q = load_ground_truth()
        for _ in tqdm.tqdm(range(num_episodes), desc=f'Computing MSE for λ={lmda}'):
            episode = Episode(strategy=sarsa_lambda_strategy)
            episode.run()
            mses.append(compute_mse(sarsa_lambda_strategy.q, ground_truth_q))
        return mses
    
    # Generate plot for final MSE per λ value.
    fig, ax = plt.subplots(figsize=(8, 6))
    lmda_mse = {}
    for lmda in SARSA_LAMBDAS:
        mses = single_run(lmda)
        lmda_mse[lmda] = mses
        ax.plot(range(1, num_episodes + 1), mses, label=f'λ={lmda}')
    ax.set_title('SARSA(λ) MSE against Monte Carlo Control Strategy')
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    fig.tight_layout()
    fig.savefig("SarsaLambdaControlStrategy_MSE.png")

    # Generate final MSE per λ value.
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    final_mses = [lmda_mse[lmda][-1] for lmda in SARSA_LAMBDAS]
    ax2.plot(SARSA_LAMBDAS, final_mses, marker='o')
    ax2.set_title('SARSA(λ) Final MSE against Monte Carlo Control Strategy')
    ax2.set_xlabel('λ Value')
    ax2.set_ylabel('Final Mean Squared Error')
    fig2.tight_layout()
    fig2.savefig("SarsaLambdaControlStrategy_Final_MSE.png")
