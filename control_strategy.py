"""Module for all available control strategies in Easy21."""

from collections import defaultdict
from enum import StrEnum
from dataclasses import dataclass

import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Hyperparameters.
N0 = 100  # Used for epsilon-greedy. episilon = N0 / (N0 + N(s))


def _initial_action_distribution() -> dict:
    """Module-level factory returning the initial action probability dict.

    Using a top-level function (instead of a local lambda) makes the
    defaultdict pickling-safe.
    """
    return {Action.HIT: 0.5, Action.STICK: 0.5}


@dataclass(frozen=True)
class State:
    '''Defines the state of the Easy21 game.'''
    player_value: int
    dealer_value: int
    # Design decision: we do not encode the status of player/dealer in the state.
    # This reduces the state space size, also making action value plots visualizable.
    # Downside is possibly slower convergence due to coarse state representations.
    # player_status: AgentStatus
    # dealer_status: AgentStatus

    def __str__(self) -> str:
        tokens = [
            str(self.player_value),
            str(self.dealer_value),
        ]
        return f'{":".join(tokens)}'


class Action(StrEnum):
    '''Defines the actions available to the player.'''
    HIT = "hit"
    STICK = "stick"


@dataclass(frozen=True)
class Policy:
    '''Defines a policy mapping states to proability distribution of actions.'''

    # Conditional probability of actions for a given state.
    distribution: dict[State, dict[Action, float]]

    def sample(self, state: State) -> Action:
        '''Samples an action based on the policy's distribution for the given state.'''
        actions = self.distribution[state].keys()
        if not actions:
            # Fallback to uniform random if this state hasn't been explored before.
            return random.choice(list(Action))
        probabilities = self.distribution[state].values()
        return random.choices(list(actions), weights=list(probabilities), k=1)[0]


class ControlStrategy:
    """Abstract base class for control strategy."""

    def next_action(self, state: State) -> Action:
        """Decides the next action based on the strategy."""
        raise NotImplementedError()

    def get_plot_df(self) -> pd.DataFrame:
        '''Returns a DataFrame for plotting the learned Q values.'''
        raise NotImplementedError()
    
    def plot_optimal_value(self) -> None:
        '''Plots the optimal value function based on learned Q values.'''
        plt.figure(figsize=(10, 6))
        plt.title(f'{self.__class__.__name__} Optimal Value Function')
        sns.heatmap(self.get_plot_df(), cmap='viridis')
        # Note that df's index (aka rows aka y axis) is player_value,
        # columns (aka x axis) is dealer_value.
        plt.xlabel('Dealer value')
        plt.ylabel('Player value')
        plt.savefig(f'{self.__class__.__name__}.png')
        plt.show()
    
    def persist(self) -> None:
        '''Persists learned results to disk.'''
        with open(f'{self.__class__.__name__}.pkl', 'wb') as f:
            pickle.dump(self, f)

class RandomControlStrategy(ControlStrategy):
    """A control strategy that selects actions randomly."""

    def next_action(self, state: State) -> Action:
        return random.choice(list(Action))


class MonteCarloControlStrategy(ControlStrategy):
    """Implements the Monte Carlo based control strategy.
    
    See slide 16 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
    """

    def __init__(self) -> None:
        # Monte Carlo control parameters.
        self.n: pd.DataFrame = pd.DataFrame()
        self.q: pd.DataFrame = pd.DataFrame()

        # Initialize policy to uniform random.
        self.policy = Policy(
            distribution=defaultdict(
                _initial_action_distribution
            )
        )

    def policy_iteration(self, trajectory: list[State|Action|int]) -> None:
        '''Updates the counters, Q values, and improved policy based on the trajectory.'''
        # Policy evaluation using every-visit Monte Carlo.
        gain = 0
        for i in range(0, len(trajectory) - 3, 3):
            state, action, reward = trajectory[i], trajectory[i+1], trajectory[i+2]
            gain += reward
            # Note: cannot do self.n.at[state, action] = 0 directly due to pandas behavior.
            # Pandas does not actually create a new row or column. Also Pandas df expects
            # column to exist before index (row) assignment.
            if action not in self.n.columns:
                self.n[action] = 0
            if state not in self.n.index:
                self.n.loc[state] = 0
            self.n.at[state, action] += 1
            assert not pd.isna(self.n.at[state, action]), \
                f"N value should not be NaN after update. State: {state}, Action: {action}"
        if action not in self.q.columns:
            self.q[action] = 0.0
        if state not in self.q.index:
            self.q.loc[state] = 0.0
        current_q = self.q.at[state, action]
        if pd.isna(current_q):
            current_q = 0.0
        alpha = 1.0 / self.n.at[state, action]
        self.q.at[state, action] += alpha * (gain - current_q)
        assert not pd.isna(self.q.at[state, action]), \
            f"Q value should not be NaN after update. State: {state}, Action: {action}"

        # Policy improvement using episilon-greedy. Based on slide 11 in
        # https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
        new_distribution = defaultdict(_initial_action_distribution)
        for state in self.q.index:
            best_action = self.q.loc[state].idxmax()
            available_actions = list(Action)
            m = len(available_actions)
            epsilon = N0 / (N0 + self.n.loc[state].max())
            new_distribution[state][best_action] = epsilon / m + 1 - epsilon
            for action in available_actions:
                new_distribution[state][action] = epsilon / m
        self.policy = Policy(distribution=new_distribution)

    def next_action(self, state: State) -> Action:
        action = self.policy.sample(state)
        return action
    
    def get_plot_df(self) -> pd.DataFrame:
        '''Plots the optimal value function based on the learned Q values.'''
        max_q = self.q.max(axis=1)
        plot_df = pd.DataFrame()
        for state in max_q.index:
            player_value, dealer_value = state.player_value, state.dealer_value
            plot_df.at[player_value, dealer_value] = max_q.at[state]
        return plot_df.sort_index().sort_index(axis=1)
