"""Module for running Easy21 episodes."""

from collections.abc import Callable
from game import Easy21
from participant import Participant, ParticipantStatus
from control_strategy import ControlStrategy, State, Action


class Episode:
    """A single run of an episode.

    This class implements the *interface* of a Markov Decsion Process (MDP), inlcuding
    taking actions, and computing rewards.

    Note that state is not explicitly modeled, because it's a composition over other
    entities' states (i.e. both player and dealer collectively defines the state of the
    MDP). That said, the state is effectively a pair of (player's value, dealer's value).
    """

    def __init__(self, strategy: ControlStrategy = ControlStrategy()) -> None:
        self.game = Easy21()
        self.player = Participant(self.game)
        self.dealer = Participant(self.game)
        self.strategy = strategy
        self.strategy.reset(
            State(
                player_value=self.player.value,
                dealer_value=self.dealer.value,
                is_terminal=False,
            )
        )

    def is_terminal(self) -> bool:
        return (
            self.player.status != ParticipantStatus.PLAYING
            or self.dealer.status != ParticipantStatus.PLAYING
        )

    def step(self, action: Action) -> None:
        """Takes one step after _player_'s action."""

        if self.is_terminal():
            return  # No further actions possible.
        match action:
            case Action.HIT:
                self.player.hit()
                reward = -1 if self.player.status == ParticipantStatus.BUSTED else 0
            case Action.STICK:
                # If player sticks, dealer takes turn. Dealer always sticks on any sum
                # of 17 or greater, and hits otherwise.
                self.player.stick()
                while self.dealer.status == ParticipantStatus.PLAYING:
                    if self.dealer.value >= 17:
                        self.dealer.stick()
                    else:
                        self.dealer.hit()
                reward = 0
                if self.dealer.status == ParticipantStatus.BUSTED:
                    reward = 1
                elif self.player.value != self.dealer.value:  # dealer sticked.
                    reward = 1 if self.player.value > self.dealer.value else -1
            case _:
                raise ValueError(f"Unknown action: {action}")
        new_state = State(
            player_value=self.player.value,
            dealer_value=self.dealer.value,
            is_terminal=self.is_terminal(),
        )
        self.strategy.observe(reward, new_state)

    def run(self) -> None:
        """Runs the episode until terminal state is reached."""
        while not self.is_terminal():
            action = self.strategy.next_action()
            self.step(action)
        self.strategy.next_action()  # One last action call to collect final reward.
