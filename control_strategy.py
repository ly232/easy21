"""Module for all available control strategies in Easy21."""

from agent import AgentStatus
from collections import defaultdict, Counter
from enum import StrEnum
from dataclasses import dataclass

import pandas as pd
import random


@dataclass(frozen=True)
class State:
    '''Defines the state of the Easy21 game.'''
    player_value: int
    dealer_value: int
    player_status: AgentStatus
    dealer_status: AgentStatus

    def __str__(self) -> str:
        tokens = [
            str(self.player_value),
            self.player_status.value,
            str(self.dealer_value),
            self.dealer_status.value,
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
    

class RandomControlStrategy(ControlStrategy):
    """A control strategy that selects actions randomly."""

    def next_action(self, state: State) -> Action:
        return random.choice(list(Action))


class MonteCarloControlStrategy(ControlStrategy):
    """Implements the Monte Carlo based control strategy.
    
    See slide 16 in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
    """

    def __init__(
            self,
            episilon: float = 0.1) -> None:
        # Monte Carlo control parameters.
        self.n: dict[tuple[State, Action], int] = Counter() 
        self.q: pd.DataFrame = pd.DataFrame()

        # Hyperparameters.
        self.episilon = episilon

        # Initialize policy to uniform random.
        self.policy = Policy(
            distribution=defaultdict(
                lambda: {Action.HIT: 0.5, Action.STICK: 0.5}
            )
        )

    def policy_iteration(self, trajectory: list[State|Action|int]) -> None:
        '''Updates the counters, Q values, and improved policy based on the trajectory.'''
        # Policy evaluation using every-visit Monte Carlo.
        gain = 0
        for i in range(0, len(trajectory) - 3, 3):
            state, action, reward = trajectory[i], trajectory[i+1], trajectory[i+2]
            gain += reward
            self.n[(state, action)] += 1
        if state not in self.q.index or action not in self.q.columns:
            self.q.at[state, action] = 0.0
        current_q = self.q.at[state, action]
        if pd.isna(current_q):
            current_q = 0.0
        self.q.at[state, action] += 1 / self.n[(state, action)] * (gain - current_q)
        
        # Policy improvement using episilon-greedy. Based on slide 11 in
        # https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf
        new_distribution = defaultdict(lambda: defaultdict(float))
        for state in self.q.index:
            best_action = self.q.loc[state].idxmax()
            available_actions = list(Action)
            m = len(available_actions)
            new_distribution[state][best_action] = self.episilon / m + 1 - self.episilon
            for action in available_actions:
                new_distribution[state][action] = self.episilon / m
        self.policy = Policy(distribution=new_distribution)

    def next_action(self, state: State) -> Action:
        action = self.policy.sample(state)
        return action
