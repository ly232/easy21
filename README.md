# Easy21

Project from David Silver's [Reinforcement Learning course @ UCL](https://davidstarsilver.wordpress.com/teaching/).

## Code Structure

```mermaid
classDiagram
    class Game {
        cards
        draw()
    }

    class Card {
        value : 1 - 10
        color : RED | BLACK
    }

    Game "1" --> "n" Card

    class Agent {
        game
        value
        stick()
        hit()
    }

    Agent "n" --> "1" Game

    class Player
    class Dealer

    Player ..|> Agent
    Dealer ..|> Agent

    class Episode {
        game
        dealer
        player
        strategy
        step()
    }
    
    Episode "1" --> "1" Game
    Episode "1" --> "1" Player
    Episode "1" --> "1" Dealer

    class ControlStrategy {
        next_action()
    }

    clsas RandomControlStrategy
    class MonteCarloControlStrategy {
        n_counter
        q_value
        policy
        policy_iteration()
    }

    RandomControlStrategy ..|> ControlStrategy
    MonteCarloControlStrategy ..|> ControlStrategy
    Episode "1" --> "1" ControlStrategy
```

## Monte Carlo Control

## TD Learning

## Linear Function Approximation
