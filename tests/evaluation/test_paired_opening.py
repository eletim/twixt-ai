"""The role swap must preserve the opening and separate terminal openings."""

from __future__ import annotations

from twixt_ai.agents.random import RandomAgent
from twixt_ai.evaluation.paired_opening import run_paired_openings
from twixt_ai.game import BoardDimensions, experiment_board


def test_role_swaps_share_exact_opening_and_resume(tmp_path):
    path = tmp_path / "pairs.jsonl"
    kwargs = dict(board=experiment_board("mini"), pairs=3,
        random_opening_moves=4, opening_seed=123, decision_seed=456, output=path)
    first = run_paired_openings(RandomAgent(), RandomAgent(), **kwargs)
    second = run_paired_openings(RandomAgent(), RandomAgent(), **kwargs)
    assert first == second
    assert len(path.read_text().splitlines()) == 3
    assert first["summary"]["unique_positions"] == 3
    for pair in first["pairs"]:
        red, black = pair["games"]
        assert red["opening_moves"] == black["opening_moves"] == pair["opening_moves"]
        assert red["start_position_hash"] == black["start_position_hash"] == pair["start_position_hash"]
        assert (red["candidate_side"], black["candidate_side"]) == ("red", "black")


def test_terminal_opening_is_excluded_from_strength_counts():
    result = run_paired_openings(RandomAgent(), RandomAgent(), board=BoardDimensions(2, 2),
        pairs=2, random_opening_moves=4, opening_seed=1, decision_seed=2)
    assert result["summary"]["opening_terminal_count"] == 2
    assert result["summary"]["games"] == 0
    assert result["summary"]["paired_score"] is None
