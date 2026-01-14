"""Module for all available control strategies in Easy21."""

from collections import defaultdict, Counter
from collections.abc import Sequence
from enum import StrEnum
from dataclasses import dataclass
from functools import cache
from itertools import product
from jaxtyping import Float
from typing import final, override

import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Hyperparameters.
N0 = 100  # Used for epsilon-greedy. episilon = N0 / (N0 + N(s))


@dataclass(frozen=True)
class State:
    '''Defines the state of the Easy21 game.'''
    player_value: int
    dealer_value: int
    is_terminal: bool
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


#
# Module-level factory methods.
# Using a top-level function (instead of a local lambda) makes the
# defaultdict pickling-safe.
#
def _initial_action_distribution() -> dict:
    return {Action.HIT: 0.5, Action.STICK: 0.5}

def _float_defaultdict() -> dict[Action, float]:
    return defaultdict(float)


class Policy:
    '''Defines a policy mapping states to proability distribution of actions.'''
    def __init__(self) -> None:
        # Links back to owner control strategy to access Q values later.
        self.strategy: 'ControlStrategy' = None

    def sample(self, state: State) -> Action:
        '''Samples an action based on the policy's distribution for the given state.
        
        Note since we do not materialize the state-action pairs, we use the linked
        function approximation strategy to estimate the Q values for all actions
        in the given state, then pick the best action with epsilon-greedy.
        '''
        action_values = {
            action: self.strategy.estimate_q(state, action)
            for action in Action
        }
        best_action = max(action_values, key=action_values.get)
        epsilon = 0.01
        if state in self.strategy.n:
            epsilon = N0 / (N0 + max(self.strategy.n[state].values()))
        m = len(Action)
        probabilities = {
            action: (epsilon / m + (1 - epsilon) if action == best_action else (epsilon / m))
            for action in Action
        }
        actions = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(actions, weights=weights, k=1)[0]


class ControlStrategy:
    """Base class for control strategy."""

    def __init__(self) -> None:
        self.policy = Policy()
        self.policy.strategy = self

        # Denotes the current trajectory. Note that if the same strategy object is reused over
        # multiple episodes, this trajectory will be reset at the beginning of each episode.
        self.trajectory: list[State|Action|int] = []

        # Common control parameters. Note that these are only populated for tabular based control
        # strategies; function approximation based strategies may choose to ignore these.
        self.q: dict[State, dict[Action, float]] = defaultdict(_float_defaultdict)
        self.n: dict[State, dict[Action, int]] = defaultdict(Counter)

    def observe(self, reward: int, new_state: State) -> None:
        """Observes the current state of the environment.
        
        By default, simply record the latest state. Subclasses should override this method
        to trigger policy iteration.
        """
        assert self.trajectory, 'Must call initialize(initial_state) first.'
        assert isinstance(self.trajectory[-1], Action), \
            'Unexpectedly called `strategy.observe()` when the latest trajectory entry ' \
            f'{self.trajectory[-1]} is not an Action.'
        self.trajectory.append(reward)
        self.trajectory.append(new_state)

    def estimate_q(self, state: State, action: Action) -> float:
        """Estimates the action-value for the given state and action.
        
        By default, looks up the materialized Q table. Subclasses using function approximation
        should override this method.
        """
        return self.q[state][action]

    @final
    def reset(self, initial_state: State) -> None:
        """Resets the control strategy with the initial state."""
        self.trajectory = [initial_state]

    @final    
    def next_action(self) -> Action:
        '''Decides the next action based on the current policy and latest observed state.
        
        Returns:
            The next Action to take, according to the current policy.

        Raises:
            AssertionError: If the latest trajectory entry is not a State.
        '''
        assert self.trajectory, 'Must call initialize(initial_state) first.'
        last_state = self.trajectory[-1]
        assert isinstance(last_state, State), \
            'Unexpectedly called `strategy.next_action()` when the latest trajectory entry ' \
            f'{last_state} is not a State.'
        action = self.policy.sample(last_state)
        self.trajectory.append(action)
        self.post_action_hook()
        return action
    
    def post_action_hook(self) -> None:
        '''A hook method called immediately after deciding the next action.

        Subclasses may override this method to implement custom behavior.
        By default, this method does nothing.
        '''
        assert isinstance(self.trajectory[-1], Action), \
            'Unexpectedly called `strategy.post_action_hook()` when the latest trajectory entry ' \
            f'{self.trajectory[-1]} is not an Action.'
        pass

    def get_plot_df(self) -> pd.DataFrame:
        '''Plots the optimal value function based on the learned Q values.'''
        max_q = {
            state: max(self.q[state].values())
            for state in self.q.keys()
        }
        plot_df = pd.DataFrame()
        for state, max_action_value in max_q.items():
            if state.is_terminal:
                continue
            player_value, dealer_value = state.player_value, state.dealer_value
            plot_df.at[player_value, dealer_value] = max_action_value
        return plot_df.sort_index().sort_index(axis=1)
    
    def plot_optimal_value(self, ax=None, show: bool = False) -> None:
        '''Plots the optimal value function based on learned Q values.

        Args:
            ax: Optional matplotlib Axes to draw into. If None, a new figure
                and axes will be created.
            show: If True, call `plt.show()` after plotting.
        '''
        plot_df = self.get_plot_df()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if plot_df.empty:
            ax.set_title(f'{self.__class__.__name__} Optimal Value Function')
            ax.set_xlabel('Dealer value')
            ax.set_ylabel('Player value')
            # Save an empty figure so test artifacts remain consistent.
            fig.tight_layout()
            fig.savefig(f'{self.__class__.__name__}.png')
            if show:
                plt.show()
            return

        # Ensure rows (player) and columns (dealer) are sorted ascending
        plot_df = plot_df.sort_index().sort_index(axis=1)

        # seaborn heatmap expects numeric indices/columns; convert if necessary
        sns.heatmap(plot_df, cmap='viridis', ax=ax, cbar=True)

        ax.set_title(f'{self.__class__.__name__} Optimal Value Function')
        ax.set_xlabel('Dealer value')
        ax.set_ylabel('Player value')

        fig.tight_layout()
        fig.savefig(f'{self.__class__.__name__}.png')
        if show:
            plt.show()
    
    def persist(self) -> None:
        '''Persists learned results to disk.'''
        with open(f'{self.__class__.__name__}.pkl', 'wb') as f:
            pickle.dump(self, f)


class MonteCarloControlStrategy(ControlStrategy):
    """Implements the Monte Carlo based control strategy.
    
    See slide 16 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.policy = Policy()
        self.policy.strategy = self

    def _policy_iteration(self) -> None:
        '''Updates the counters, Q values, and improved policy based on the trajectory.'''
        # Policy evaluation using every-visit Monte Carlo.
        trajectory = self.trajectory
        gain = trajectory[-2]  # for easy21, only the final reward is non-zero.
        for i in range(0, len(trajectory) - 3, 3):
            state, action = trajectory[i], trajectory[i+1]
            q = self.q[state][action]
            alpha = 1.0 / self.n[state][action]
            self.q[state][action] += alpha * (gain - q)

    @override
    def observe(self, reward: int, new_state: State) -> None:
        """Update control parameters for terminal states only."""
        super().observe(reward, new_state)

        if not new_state.is_terminal:
            return
        # Apply MC policy iteration on completed episode.
        self._policy_iteration()

    @override
    def post_action_hook(self) -> None:
        super().post_action_hook()
        last_state, last_action = self.trajectory[-2], self.trajectory[-1]
        self.n[last_state][last_action] += 1


class SarsaLambdaControlStrategy(ControlStrategy):
    """Implements the Sarsa(λ) based control strategy.
    
    See slide 29 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
    Note that we're implementing the backward view of Sarsa(λ).
    """

    def __init__(self, lmda: float) -> None:
        super().__init__()
        self.policy = Policy()
        self.policy.strategy = self

        # Sarsa(λ) control parameters.
        self.lmda = lmda
        self.e: dict[State, dict[Action, float]] = defaultdict(_float_defaultdict)
        self.q: dict[State, dict[Action, float]] = defaultdict(_float_defaultdict)

    def _policy_iteration(self) -> None:
        """Run one round of Sarsa(λ) policy iteration based on the latest trajectory."""        
        old_state, old_action, reward, new_state, new_action = \
            self.trajectory[-5], self.trajectory[-4], self.trajectory[-3], \
                self.trajectory[-2], self.trajectory[-1]
        delta = reward + self.q[new_state][new_action] - self.q[old_state][old_action]
        self.e[old_state][old_action] += 1
        # Note: unlike MC which uses alpha = 1.0 / self.n[new_state][new_action], for TD-based
        # methods we use a fixed small alpha. This is because TD has lower variance and higher
        # bias, so diminishing step size risks not correcting the bias in later episodes. OTOH,
        # MC is more suited for dinimishing step size because it suffers more from variance
        # than bias, and as MC sees more episodes, sample size increases thus stablizes the
        # variance more, at which point smaller step size is better to avoid overshooting.
        alpha = 0.01
        # Performance optimization trick: rather than iterating over the entire state-action
        # space in the action-value tabular q (which monotonically grows as more episodes are run),
        # we only need to update those state-action pairs that have non-zero eligibility trace,
        # as the eligibility trace value is multiplied in the update formula. In practice, this gives
        # roughly 7x speed up.
        eligible_state_actions = [
            (state, action)
            for state in self.e
            for action in self.e[state]
            if self.e[state][action] > 0
        ]
        for state, action in eligible_state_actions:
            self.q[state][action] += alpha * delta * self.e[state][action]
            self.e[state][action] *= self.lmda

        # ATTN: Eligibility trace does not carry over to next episode.
        if new_state.is_terminal:
            self.e = defaultdict(_float_defaultdict)

    @override
    def post_action_hook(self) -> None:
        super().post_action_hook()
        last_state, last_action = self.trajectory[-2], self.trajectory[-1]
        self.n[last_state][last_action] += 1
        # Apply Sarsa(λ) policy iteration if trajectory has at least one round of
        # (s, a, r, s', a')
        if len(self.trajectory) >= 5:
            self._policy_iteration()

class LinearFunctionApproximationSarsaLambdaControlStrategy(ControlStrategy):
    """Implements a control strategy using linear function approximation.
    
    This uses the Sarsa(λ) algorithm to *estimate* for the true target, then it uses
    linear function approximation to *approximate* the action-value function. Note
    that the action-value is never materialized into a table (this is the whoe point
    of function approximation after all). The Sarsa(λ) algorithm relies on the function
    approximation to estimate its action values with weights at the current step.
    """

    def __init__(self, lmda: float) -> None:
        super().__init__()
        self.policy = Policy()
        self.policy.strategy = self
        self.lmda = lmda

        # Unlike MC and Sarsa, we don't explicitly materialize tables for Q and N for function
        # approximation strategy. Instead we use a weight vector to estimate action-value function,
        # where the weights are trained via gradient descent using a linear model against one-hot
        # encoded feature vectors (see self._feature_vector below).
        self.weights: Float[np.ndarray, '36'] = np.zeros(36)

        # Eligibility trace vector. Note that unlike the tabular Sarsa(λ) where eligibility tracks the
        # entire state-action space, here because we mapped the |S|*|A| space into a fixed 36-dimensional
        # vector space, we just need to track the eligibility in that vector space.
        self.e: Float[np.ndarray, '36'] = np.zeros(36)

    @cache
    def _feature_vector(self, state: State, action: Action) -> Float[np.ndarray, '36']:
        """Generates the feature vector for the given state and action."""
        dealer_cuboids = [(1, 4), (4, 7), (7, 10)]
        player_cuboids = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
        features = list(product(dealer_cuboids, player_cuboids, Action))
        feature_vector = np.zeros(len(features), dtype=float)
        for i, feature in enumerate(features):
            (d_start, d_end), (p_start, p_end), a = feature
            if d_start <= state.dealer_value <= d_end \
                and p_start <= state.player_value <= p_end \
                    and a == action:
                feature_vector[i] = 1.0
        return feature_vector

    @override
    def estimate_q(self, state: State, action: Action) -> float:
        """Estimates the action-value for the given state and action using current weights."""
        phi = self._feature_vector(state, action)
        return float(np.dot(self.weights, phi))

    @override
    def post_action_hook(self) -> None:
        # Similar to Sarsa(λ), function approximation update weights after each action.
        super().post_action_hook()
        if len(self.trajectory) >= 5:
            self._policy_iteration()

    def _policy_iteration(self) -> None:
        """Run one round of policy iteration using linear function approximation.
        
        See slide 24 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-6-value-function-approximation-.pdf.
        We implement the backward view of Sarsa(λ) with eligibility traces.
        """
        old_state, old_action, reward, new_state, new_action = \
            self.trajectory[-5], self.trajectory[-4], self.trajectory[-3], \
                self.trajectory[-2], self.trajectory[-1]
        delta = reward + self.estimate_q(new_state, new_action) - self.estimate_q(old_state, old_action)
        grad = self._feature_vector(old_state, old_action)
        self.e = self.lmda * self.e + grad
        alpha = 0.01  # Fixed small step size for TD methods.
        delta_w = alpha * delta * self.e
        # print(delta_w)
        self.weights += delta_w

        # ATTN: Eligibility trace does not carry over to next episode.
        if new_state.is_terminal:
            self.e = np.zeros(36)
