'''Module for running Easy21 episodes.'''

from enum import StrEnum
from dataclasses import dataclass, field

from game import Easy21
from agent import Agent, AgentStatus

class Action(StrEnum):
    '''Defines the actions available to the player.'''
    HIT = "hit"
    STICK = "stick"


@dataclass
class Trajectory:
    '''Models the trajectory of an episode.'''
    states: list[tuple[int, int]] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    rewards: list[int] = field(default_factory=list)


class Episode:
    '''A single run of an episode.
    
    This class implements the *interface* of a Markov Decsion Process (MDP), inlcuding
    taking actions, and computing rewards.

    Note that state is not explicitly modeled, because it's a composition over other
    entities' states (i.e. both player and dealer collectively defines the state of the
    MDP). That said, the state is effectively a pair of (player's value, dealer's value).
    '''
    def __init__(self):
        self.game = Easy21()
        self.player = Agent(self.game)
        self.dealer = Agent(self.game)

        self.trajectory = Trajectory()
        self._update_state_trajectory()  # initial state

    def _update_state_trajectory(self) -> None:
        self.trajectory.states.append((self.player.value, self.dealer.value))

    def _update_action_trajectory(self, action: Action) -> None:
        self.trajectory.actions.append(action)

    def _update_reward_trajectory(self, reward: int) -> None:
        self.trajectory.rewards.append(reward)

    def is_terminal(self) -> bool:
        return self.player.status != AgentStatus.PLAYING \
            or self.dealer.status != AgentStatus.PLAYING

    def step(self, action: Action):
        if self.is_terminal():
            return  # No further actions possible.
        self._update_action_trajectory(action)
        match action:
            case Action.HIT:
                self.player.hit()
                reward = -1 if self.player.status == AgentStatus.BUSTED else 0
                self._update_reward_trajectory(reward)
            case Action.STICK:
                # If player sticks, dealer takes turn. Dealer always sticks on any sum
                # of 17 or greater, and hits otherwise.
                self.player.stick()
                while self.dealer.status == AgentStatus.PLAYING:
                    if self.dealer.value >= 17:
                        self.dealer.stick()
                    else:
                        self.dealer.hit()
                reward = 0
                if self.dealer.status == AgentStatus.BUSTED:
                    reward = 1
                elif self.player.value != self.dealer.value:  # dealer sticked.
                    reward = 1 if self.player.value > self.dealer.value else -1
                self._update_reward_trajectory(reward)
            case _:
                raise ValueError(f"Unknown action: {action}")

    def state_value(self) -> int:
        ...

    def action_value(self, action: Action) -> int:
        ...
