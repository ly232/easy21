'''Module for running Easy21 episodes.'''

from game import Easy21
from agent import Agent, AgentStatus
from control_strategy import ControlStrategy, RandomControlStrategy, State, Action


class Episode:
    '''A single run of an episode.
    
    This class implements the *interface* of a Markov Decsion Process (MDP), inlcuding
    taking actions, and computing rewards.

    Note that state is not explicitly modeled, because it's a composition over other
    entities' states (i.e. both player and dealer collectively defines the state of the
    MDP). That said, the state is effectively a pair of (player's value, dealer's value).
    '''
    def __init__(self, strategy: ControlStrategy=RandomControlStrategy()) -> None:
        self.game = Easy21()
        self.player = Agent(self.game)
        self.dealer = Agent(self.game)
        self.strategy = strategy

        self.trajectory: list[State|Action|int] = [
            State(
                player_value=self.player.value,
                dealer_value=self.dealer.value,
                player_status=self.player.status,
                dealer_status=self.dealer.status
            )
        ]

    def is_terminal(self) -> bool:
        return self.player.status != AgentStatus.PLAYING \
            or self.dealer.status != AgentStatus.PLAYING

    def step(self, action: Action):
        """Takes one step after _player_'s action."""

        if self.is_terminal():
            return  # No further actions possible.
        self.trajectory.append(action)
        match action:
            case Action.HIT:
                self.player.hit()
                reward = -1 if self.player.status == AgentStatus.BUSTED else 0
                self.trajectory.append(reward)
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
                self.trajectory.append(reward)
            case _:
                raise ValueError(f"Unknown action: {action}")
        new_state = State(
            player_value=self.player.value,
            dealer_value=self.dealer.value,
            player_status=self.player.status,
            dealer_status=self.dealer.status
        )
        self.trajectory.append(new_state)

    def run(self) -> list[State|Action|int]:
        '''Runs the episode until terminal state is reached.
        
        Returns the trajectory of the episode.
        '''
        while not self.is_terminal():
            state = State(
                player_value=self.player.value,
                dealer_value=self.dealer.value,
                player_status=self.player.status,
                dealer_status=self.dealer.status
            )
            action = self.strategy.next_action(state)
            self.step(action)
        return self.trajectory
