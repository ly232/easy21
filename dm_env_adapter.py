"""Adapts Easy21 implementations to conform with dm_env.Environment interface."""

from typing import override

import control_strategy
import episode

import dm_env
from dm_env import specs

import numpy as np


class DeepmindRlAgent:
    """An RL agent that interacts with dm_env.Environment."""

    def __init__(self, strategy: control_strategy.ControlStrategy) -> None:
        self._strategy: control_strategy.ControlStrategy = strategy

    def step(self, timestep: dm_env.TimeStep) -> control_strategy.Action:
        """Selects an action based on the current timestep.

        Args:
            timestep: The current timestep from the environment. Unused; only here
                to conform with typical dm_env agent interface as documented in
                dm_env docs.
        """
        return self._strategy.next_action()


class DeepmindEnvironmentAdapter(dm_env.Environment):
    """Adapts Easy21 to dm_env.Environment interface."""

    def __init__(self, strategy: control_strategy.ControlStrategy) -> None:
        self._strategy: control_strategy.ControlStrategy = strategy
        self._episode: episode.Episode = None

    @override
    def reset(self) -> dm_env.TimeStep:
        self._episode = episode.Episode(strategy=self._strategy)
        return dm_env.restart(
            [
                self._episode.player.value,
                self._episode.dealer.value,
            ]
        )

    @override
    def step(self, action: control_strategy.Action) -> dm_env.TimeStep:
        self._episode.step(action)
        if self._episode.is_terminal():
            reward = 0
            if self._episode.player.status == episode.ParticipantStatus.BUSTED:
                reward = -1
            elif (
                self._episode.dealer.status == episode.ParticipantStatus.BUSTED
                or self._episode.player.value > self._episode.dealer.value
            ):
                reward = 1
            return dm_env.termination(
                reward=reward,
                observation=[
                    self._episode.player.value,
                    self._episode.dealer.value,
                ],
            )
        else:
            return dm_env.transition(
                reward=0,
                observation=[
                    self._episode.player.value,
                    self._episode.dealer.value,
                ],
            )

    @override
    def observation_spec(self):
        """Observation spec defines the data model for states."""
        return specs.BoundedArray(
            shape=(2,),  # player_value, dealer_value
            dtype=np.int32,
            name="player_dealer_values",
            minimum=np.array([1, 1], dtype=np.int32),
            # Player can go up to 21, dealer up to 10 because it only
            # reflects dealer's first card. After player sticks or busts,
            # dealer will keep playing till termination, and dealer's final
            # value is not part of the state representation (we capture final
            # reward instead).
            maximum=np.array([21, 10], dtype=np.int32),
        )

    @override
    def action_spec(self):
        """Action spec defines the data model for actions."""
        return specs.DiscreteArray(
            dtype=np.int32,
            num_values=len(control_strategy.Action),
            name="easy21_action",
        )
