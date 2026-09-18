"""Role-swapped evaluation from identical uniformly sampled openings."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random

from twixt_ai.evaluation.match import MatchConfig, run_match
from twixt_ai.game import BoardDimensions, Player, apply_move, create_game


def opening_position(board: BoardDimensions, moves: tuple) -> object:
    state = create_game(board)
    for move in moves:
        state = apply_move(state, move)
    return state


def position_hash(state: object) -> str:
    return hashlib.sha256(json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize(pairs: list[dict]) -> dict:
    usable = [pair for pair in pairs if not pair["opening_terminal"]]
    games = [game for pair in usable for game in pair["games"]]
    def wdl(items: list[dict]) -> dict[str, int]:
        return {key: sum(game["result"] == key for game in items) for key in ("win", "draw", "loss")}
    scores = [sum({"win": 1.0, "draw": 0.5, "loss": 0.0}[g["result"]] for g in pair["games"]) for pair in usable]
    hashes = Counter(pair["start_position_hash"] for pair in pairs)
    return {
        "pairs": len(pairs), "evaluated_pairs": len(usable),
        "opening_terminal_count": len(pairs) - len(usable),
        "games": len(games), "overall": wdl(games),
        "red": wdl([g for g in games if g["candidate_side"] == "red"]),
        "black": wdl([g for g in games if g["candidate_side"] == "black"]),
        "paired_score": sum(scores) / (2 * len(scores)) if scores else None,
        "better_pairs": sum(score > 1 for score in scores),
        "equal_pairs": sum(score == 1 for score in scores),
        "worse_pairs": sum(score < 1 for score in scores),
        "both_win": sum(score == 2 for score in scores),
        "split": sum(sorted(g["result"] for g in pair["games"]) == ["loss", "win"] for pair in usable),
        "both_loss": sum(score == 0 for score in scores),
        "average_move_count": sum(g["move_count"] for g in games) / len(games) if games else None,
        "draw_rate": sum(g["result"] == "draw" for g in games) / len(games) if games else None,
        "red_win_rate": sum(g["winner"] == "red" for g in games) / len(games) if games else None,
        "unique_positions": len(hashes),
        "largest_position_multiplicity": max(hashes.values(), default=0),
    }


def run_paired_openings(
    candidate, opponent, *, board: BoardDimensions, pairs: int,
    random_opening_moves: int, opening_seed: int, decision_seed: int,
    output: Path | None = None, candidate_name: str = "candidate",
    opponent_name: str = "frozen_gen11",
) -> dict:
    """Save each completed pair so a long run can be resumed safely."""
    if pairs < 1 or random_opening_moves < 0:
        raise ValueError("pairs must be positive and opening length non-negative")
    seed_source = Random(opening_seed)
    decision_source = Random(decision_seed)
    records: list[dict] = []
    if output is not None and output.exists():
        records = [json.loads(line) for line in output.read_text().splitlines() if line]
        if len(records) > pairs:
            raise ValueError("saved pair count exceeds requested pairs")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(pairs):
        pair_opening_seed = seed_source.getrandbits(64)
        pair_decision_seed = decision_source.getrandbits(64)
        if index < len(records):
            if records[index]["opening_seed"] != pair_opening_seed or records[index]["decision_seed"] != pair_decision_seed:
                raise ValueError("saved pair seed does not match requested suite")
            continue
        pair_games = []
        for side in (Player.RED, Player.BLACK):
            red, black = (candidate, opponent) if side is Player.RED else (opponent, candidate)
            match = run_match(red, black, config=MatchConfig(board, pair_decision_seed,
                candidate_name if side is Player.RED else opponent_name,
                opponent_name if side is Player.RED else candidate_name),
                random_opening_moves=random_opening_moves, opening_seed=pair_opening_seed)
            opening_moves = match.moves[:min(random_opening_moves, len(match.moves))]
            start = opening_position(board, opening_moves)
            opening = [{"player": move.player.value, "coordinate": move.coordinate.to_dict()} for move in opening_moves]
            if pair_games and (opening != pair_games[0]["opening_moves"] or position_hash(start) != pair_games[0]["start_position_hash"]):
                raise AssertionError("role-swapped games diverged during opening")
            result = "draw" if match.winner is None else "win" if match.winner is side else "loss"
            pair_games.append({"candidate_side": side.value, "winner": match.winner.value if match.winner else None,
                "result": result, "move_count": len(match.moves), "opening_moves": opening,
                "start_position_hash": position_hash(start)})
        record = {"pair_index": index, "opening_seed": pair_opening_seed,
            "decision_seed": pair_decision_seed, "opening_moves": pair_games[0]["opening_moves"],
            "start_position_hash": pair_games[0]["start_position_hash"],
            "side_to_move": start.side_to_move.value, "opening_terminal": start.is_terminal,
            "opening_result": start.result.value, "games": pair_games}
        records.append(record)
        if output is not None:
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    return {"format": "twixt-ai-paired-opening", "version": 1,
        "config": {"candidate": candidate_name, "opponent": opponent_name, "board": board.to_dict(),
            "pairs": pairs, "random_opening_moves": random_opening_moves,
            "opening_seed": opening_seed, "decision_seed": decision_seed},
        "summary": summarize(records), "pairs": records}
