"""Module for all available control strategies in Easy21."""

from collections import defaultdict, Counter
from collections.abc import Sequence
from enum import StrEnum
from dataclasses import dataclass
from typing import final, override

import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
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

    # Conditional probability of actions for a given state.
    # Note: cannot use lambda because of pickling issues.
    distribution: dict[State, dict[Action, float]] = defaultdict(_initial_action_distribution)

    def sample(self, state: State) -> Action:
        '''Samples an action based on the policy's distribution for the given state.'''
        if state not in self.distribution:
            # Fallback to uniform random if this state hasn't been explored before.
            return random.choice(list(Action))
        actions = list(self.distribution[state].keys())
        probabilities = list(self.distribution[state].values())
        return random.choices(actions, weights=list(probabilities), k=1)[0]
    
    def update(
            self, 
            states: Sequence[State], 
            best_action: Action, 
            epsilon: float) -> None:
        '''Updates the policy's distribution using epsilon-greedy.'''
        new_distribution = defaultdict(_initial_action_distribution)
        for state in states:
            available_actions = list(Action)
            m = len(available_actions)
            new_distribution[state][best_action] = epsilon / m + 1 - epsilon
            for action in available_actions:
                new_distribution[state][action] = epsilon / m
        self.distribution = new_distribution


class ControlStrategy:
    """Base class for control strategy."""

    def __init__(self) -> None:
        # Denotes the current trajectory. Note that if the same strategy object is reused over
        # multiple episodes, this trajectory will be reset at the beginning of each episode.
        self.trajectory: list[State|Action|int] = []

        # Common control parameters.
        self.q: dict[State, dict[Action, float]] = defaultdict(_float_defaultdict)
        self.n: dict[State, dict[Action, int]] = defaultdict(Counter)

        # Initialize policy to uniform random.
        self.policy = Policy()

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
        self.n[last_state][action] += 1
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

        # Policy improvement using episilon-greedy. Based on slide 11 in
        # https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
        self.policy.update(
            self.q.keys(), 
            best_action=max(self.q[state], key=self.q[state].get), 
            epsilon=N0 / (N0 + max(self.n[state].values()))
        )

    @override
    def observe(self, reward: int, new_state: State) -> None:
        """Update control parameters for terminal states only."""
        super().observe(reward, new_state)

        if not new_state.is_terminal:
            return
        # Apply MC policy iteration on completed episode.
        self._policy_iteration()


class SarsaLambdaControlStrategy(ControlStrategy):
    """Implements the Sarsa(λ) based control strategy.
    
    See slide 29 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
    Note that we're implementing the backward view of Sarsa(λ).
    """

    def __init__(self, lmda: float) -> None:
        super().__init__()

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
        alpha = 1.0 / self.n[new_state][new_action]
        for state in self.q.keys():
            for action in self.q[state].keys():
                self.q[state][action] += alpha * delta * self.e[state][action]
                self.e[state][action] *= self.lmda

    @override
    def post_action_hook(self) -> None:
        super().post_action_hook()
        # Apply Sarsa(λ) policy iteration if trajectory has at least one round of
        # (s, a, r, s', a')
        if len(self.trajectory) >= 5:
            self._policy_iteration()
