"""Small deterministic strength check for the first competitive agent."""

from twixt_ai.agents import RandomAgent
from twixt_ai.evaluation import run_match
from twixt_ai.game import BoardDimensions, Player
from twixt_ai.search import SearchAgent


def test_heuristic_search_materially_beats_random() -> None:
    """Search should win at least 75% while playing both sides equally."""

    board = BoardDimensions(6, 6)
    wins = 0
    games = 0
    for seed in range(6):
        for search_side in (Player.RED, Player.BLACK):
            search = SearchAgent()
            random = RandomAgent()
            result = run_match(
                search if search_side is Player.RED else random,
                random if search_side is Player.RED else search,
                board=board,
                seed=seed,
            )
            games += 1
            wins += result.winner is search_side

    assert wins / games >= 0.75
