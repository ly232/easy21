from agent import Agent, AgentStatus
from game import Easy21

import pytest

def test_agent_stick(subtests: pytest.SubTests) -> None:
    game = Easy21()
    agent = Agent(game)

    with subtests.test('Initial status'):
        assert 1 <= agent.value <= 10
        assert agent.status == AgentStatus.PLAYING

    with subtests.test('Stick action'):
        agent.stick()
        assert agent.status == AgentStatus.STICKED
        with pytest.raises(AssertionError):
            agent.stick()
        with pytest.raises(AssertionError):
            agent.hit()

def test_agent_hit(subtests: pytest.SubTests) -> None:
    game = Easy21()
    agent = Agent(game)

    with subtests.test('Initial status'):
        assert 1 <= agent.value <= 10
        assert agent.status == AgentStatus.PLAYING

    with subtests.test('Hit action leading to bust'):
        while agent.status == AgentStatus.PLAYING:
            agent.hit()
        assert agent.status == AgentStatus.BUSTED
        assert agent.value < 1 or agent.value > 21