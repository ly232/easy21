from participant import Participant, ParticipantStatus
from game import Easy21

import pytest


def test_participant_stick(subtests: pytest.SubTests) -> None:
    game = Easy21()
    participant = Participant(game)

    with subtests.test("Initial status"):
        assert 1 <= participant.value <= 10
        assert participant.status == ParticipantStatus.PLAYING

    with subtests.test("Stick action"):
        participant.stick()
        assert participant.status == ParticipantStatus.STICKED
        with pytest.raises(AssertionError):
            participant.stick()
        with pytest.raises(AssertionError):
            participant.hit()


def test_participant_hit(subtests: pytest.SubTests) -> None:
    game = Easy21()
    participant = Participant(game)

    with subtests.test("Initial status"):
        assert 1 <= participant.value <= 10
        assert participant.status == ParticipantStatus.PLAYING

    with subtests.test("Hit action leading to bust"):
        while participant.status == ParticipantStatus.PLAYING:
            participant.hit()
        assert participant.status == ParticipantStatus.BUSTED
        assert participant.value < 1 or participant.value > 21
