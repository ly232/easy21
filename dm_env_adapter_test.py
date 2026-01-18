"""Unit tests for dm_env_adapter.py

uv run pytest dm_env_adapter_test.py::test_deepmind_rl_api_adapter_integration -s
"""

import dm_env_adapter
import control_strategy
import tqdm

import pytest


_NUM_EPISODES = 100000

_CONTROL_STRATEGIES = [
    control_strategy.MonteCarloControlStrategy(),
    control_strategy.SarsaLambdaControlStrategy(lmda=0.5),
    control_strategy.LinearFunctionApproximationSarsaLambdaControlStrategy(lmda=0.5),
    control_strategy.MonteCarloPolicyGradientControlStrategy(),
    control_strategy.ActorCriticControlStrategy(
        control_strategy.LinearFunctionApproximationSarsaLambdaControlStrategy(lmda=0.5)
    ),
]


@pytest.mark.parametrize("strategy", _CONTROL_STRATEGIES)
def test_deepmind_rl_api_adapter_integration(
    strategy: control_strategy.ControlStrategy,
) -> None:
    agent = dm_env_adapter.DeepmindRlAgent(strategy)
    env = dm_env_adapter.DeepmindEnvironmentAdapter(strategy)
    timestep = env.reset()
    for _ in tqdm.tqdm(range(_NUM_EPISODES)):
        while not timestep.last():
            action = agent.step(timestep)
            timestep = env.step(action)
        agent.step(timestep)  # Final step to observe reward.
        timestep = env.reset()
    strategy.plot_optimal_value(
        show=False,
        save_fig=True,
        fig_name=f"figures/Deepmind_RL_Env_{strategy.__class__.__name__}.png",
    )
