"""Headless orchestration for complete agent-versus-agent matches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from random import Random
from types import MappingProxyType
from typing import Mapping

from twixt_ai.agents import Agent, AgentResult, select_agent_move
from twixt_ai.game import (
    BoardDimensions,
    GameRecord,
    GameResult,
    GameState,
    PegPlacement,
    Player,
    apply_move,
    create_game,
)


MATCH_FORMAT = "twixt-ai-match"
MATCH_FORMAT_VERSION = 1


def _require_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Serializable inputs needed to identify and reproduce a match."""

    board: BoardDimensions = BoardDimensions()
    seed: int | None = None
    red_agent: str = "red"
    black_agent: str = "black"

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        _require_seed(self.seed)
        if not isinstance(self.red_agent, str) or not self.red_agent:
            raise ValueError("red_agent must be a non-empty string")
        if not isinstance(self.black_agent, str) or not self.black_agent:
            raise ValueError("black_agent must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_dict(),
            "seed": self.seed,
            "agents": {
                Player.RED.value: self.red_agent,
                Player.BLACK.value: self.black_agent,
            },
        }


@dataclass(frozen=True, slots=True)
class MatchDecision:
    """One validated agent decision and the input seed used to make it."""

    move: PegPlacement
    seed: int | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.move, PegPlacement):
            raise TypeError("move must be a PegPlacement")
        _require_seed(self.seed)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_agent_result(cls, result: AgentResult, seed: int | None) -> MatchDecision:
        return cls(move=result.move, seed=seed, metadata=result.metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "player": self.move.player.value,
            "coordinate": self.move.coordinate.to_dict(),
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A complete match artifact suitable for evaluation and persistence."""

    config: MatchConfig
    record: GameRecord
    decisions: tuple[MatchDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, MatchConfig):
            raise TypeError("config must be a MatchConfig")
        if not isinstance(self.record, GameRecord):
            raise TypeError("record must be a GameRecord")
        decisions = tuple(self.decisions)
        if any(not isinstance(item, MatchDecision) for item in decisions):
            raise TypeError("decisions must contain only MatchDecision values")
        if tuple(item.move for item in decisions) != self.record.moves:
            raise ValueError("decisions must describe the recorded moves")
        if self.record.initial_state.board != self.config.board:
            raise ValueError("record board must match match configuration")
        if not self.record.final_state.is_terminal:
            raise ValueError("a match result must contain a terminal game record")
        object.__setattr__(self, "decisions", decisions)

    @property
    def final_state(self) -> GameState:
        return self.record.final_state

    @property
    def game_result(self) -> GameResult:
        return self.final_state.result

    @property
    def winner(self) -> Player | None:
        return self.final_state.winner

    @property
    def moves(self) -> tuple[PegPlacement, ...]:
        return self.record.moves

    def to_dict(self) -> dict[str, object]:
        return {
            "format": MATCH_FORMAT,
            "version": MATCH_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "record": self.record.to_dict(),
            "result": {
                "status": self.game_result.value,
                "winner": self.winner.value if self.winner is not None else None,
                "move_count": len(self.moves),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this artifact with deterministic key ordering."""

        separators = None if indent is not None else (",", ":")
        return json.dumps(self.to_dict(), sort_keys=True, separators=separators, indent=indent)


def run_match(
    red_agent: Agent,
    black_agent: Agent,
    *,
    config: MatchConfig | None = None,
    board: BoardDimensions | None = None,
    seed: int | None = None,
) -> MatchResult:
    """Run both assigned agents until the canonical engine ends the game.

    When ``config.seed`` is set, it deterministically generates a distinct
    decision-local seed for every turn. Passing ``None`` preserves the agent
    interface's explicitly unseeded behavior.
    """

    if config is not None and not isinstance(config, MatchConfig):
        raise TypeError("config must be a MatchConfig or None")
    if config is not None and (board is not None or seed is not None):
        raise ValueError("board and seed cannot override an explicit config")
    if board is not None and not isinstance(board, BoardDimensions):
        raise TypeError("board must be BoardDimensions or None")
    _require_seed(seed)

    match_config = config or MatchConfig(
        board=board or BoardDimensions(),
        seed=seed,
        red_agent=type(red_agent).__name__,
        black_agent=type(black_agent).__name__,
    )

    agents = {Player.RED: red_agent, Player.BLACK: black_agent}
    for player, agent in agents.items():
        if not isinstance(agent, Agent):
            raise TypeError(f"{player.value}_agent must implement choose_move(request)")

    initial_state = create_game(match_config.board)
    state = initial_state
    decisions: list[MatchDecision] = []
    seed_source = Random(match_config.seed) if match_config.seed is not None else None

    while not state.is_terminal:
        decision_seed = seed_source.randrange(2**64) if seed_source is not None else None
        result = select_agent_move(agents[state.side_to_move], state, seed=decision_seed)
        decisions.append(MatchDecision.from_agent_result(result, decision_seed))
        state = apply_move(state, result.move)

    record = GameRecord(initial_state, tuple(item.move for item in decisions), state)
    return MatchResult(match_config, record, tuple(decisions))


__all__ = [
    "MATCH_FORMAT",
    "MATCH_FORMAT_VERSION",
    "MatchConfig",
    "MatchDecision",
    "MatchResult",
    "run_match",
]
