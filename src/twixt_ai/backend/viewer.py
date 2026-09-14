"""Developer-facing AI-vs-AI match generation and replay projection."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Mapping

from twixt_ai.agents import Agent
from twixt_ai.evaluation.match import MatchConfig, MatchResult, run_match
from twixt_ai.game import BoardDimensions, GameRecord, Player, apply_move
from twixt_ai.search import DEFAULT_ROLLOUT_LIMIT, MCTSAgent


VIEWER_AGENT_MODES = (
    "non-neural-mcts",
    "learned-policy-only",
    "learned-value-only",
    "learned-policy-value",
)
DEFAULT_VIEWER_SIMULATIONS = 20
MAX_LISTED_ARTIFACTS = 200


class ViewerService:
    """Build replay data without adding presentation concerns to game/search code."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        *,
        simulations: int = DEFAULT_VIEWER_SIMULATIONS,
        rollout_limit: int = DEFAULT_ROLLOUT_LIMIT,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.simulations = simulations
        self.rollout_limit = rollout_limit
        self._neural_cache: dict[str, object] = {}
        self._lock = Lock()

    def _paths(self, pattern: str) -> list[Path]:
        experiments = self.workspace_root / "experiments"
        return sorted(path for path in experiments.glob(pattern) if path.is_file())

    def _checkpoint_map(self) -> dict[str, Path]:
        return {
            path.relative_to(self.workspace_root).as_posix(): path
            for path in self._paths("**/best.pt")
        }

    def _artifact_map(self) -> dict[str, Path]:
        paths = self._paths("**/games/game-*.json")
        # Keep the selector useful across experiments instead of letting one
        # large 5k run consume every slot. Sample the tail of every games/
        # collection, where interrupted/resumed experiment output also lands.
        collections: dict[Path, list[Path]] = {}
        for path in paths:
            collections.setdefault(path.parent, []).append(path)
        quota = max(1, MAX_LISTED_ARTIFACTS // max(1, len(collections)))
        selected = sorted(
            path
            for collection in collections.values()
            for path in collection[-quota:]
        )[-MAX_LISTED_ARTIFACTS:]
        return {
            path.relative_to(self.workspace_root).as_posix(): path
            for path in selected
        }

    def configuration(self) -> dict[str, object]:
        checkpoints = list(self._checkpoint_map())
        artifacts = list(self._artifact_map())
        return {
            "board": {"preset": "mini", "width": 10, "height": 10},
            "agent_modes": list(VIEWER_AGENT_MODES),
            "checkpoints": [
                {"id": item, "label": item.removeprefix("experiments/")}
                for item in checkpoints
            ],
            "artifacts": [
                {"id": item, "label": item.removeprefix("experiments/")}
                for item in artifacts
            ],
            "artifact_limit": MAX_LISTED_ARTIFACTS,
            "artifact_total": len(self._paths("**/games/game-*.json")),
            "search": {
                "simulations": self.simulations,
                "rollout_limit": self.rollout_limit,
            },
        }

    def _agent(self, mode: str, checkpoint_id: object) -> Agent:
        if mode not in VIEWER_AGENT_MODES:
            raise ValueError("unknown viewer agent mode")
        if mode == "non-neural-mcts":
            return MCTSAgent(
                simulations=self.simulations,
                rollout_limit=self.rollout_limit,
            )
        if not isinstance(checkpoint_id, str):
            raise ValueError("learned agents require a checkpoint")
        checkpoint = self._checkpoint_map().get(checkpoint_id)
        if checkpoint is None:
            raise ValueError("unknown checkpoint")

        # Keep heavyweight model dependencies and checkpoint loading out of the
        # Human-vs-AI path. A loaded inference adapter is safely reused by the
        # synchronous development server; each player still gets its own tree.
        from twixt_ai.device import select_device
        from twixt_ai.evaluation.mini_strength import AblatedPolicyValue
        from twixt_ai.models import load_policy_value_checkpoint
        from twixt_ai.search.neural import NeuralPolicyValue

        with self._lock:
            neural = self._neural_cache.get(checkpoint_id)
            if neural is None:
                device = select_device("auto")
                loaded = load_policy_value_checkpoint(
                    checkpoint, map_location=device.resolved_device
                )
                if (
                    loaded.model.config.board_width != 10
                    or loaded.model.config.board_height != 10
                ):
                    raise ValueError("checkpoint must target the Mini 10x10 board")
                neural = NeuralPolicyValue(loaded.model)
                self._neural_cache[checkpoint_id] = neural
        guidance_mode = mode.removeprefix("learned-")
        return MCTSAgent(
            simulations=self.simulations,
            rollout_limit=self.rollout_limit,
            policy_value=AblatedPolicyValue(neural, guidance_mode),  # type: ignore[arg-type]
        )

    @staticmethod
    def _board_capacity(board: BoardDimensions) -> int:
        # The four corner holes are forbidden to both players in Twixt.
        return max(0, board.width * board.height - 4)

    def _replay(
        self,
        match: MatchResult,
        *,
        source: Mapping[str, object],
        agent_settings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        state = match.record.initial_state
        frames: list[dict[str, object]] = [
            {"move": 0, "state": state.to_dict(), "last_move": None, "decision": None}
        ]
        for move_number, decision in enumerate(match.decisions, 1):
            state = apply_move(state, decision.move)
            # MatchDecision owns the recursive JSON thawing contract; a
            # shallow dict() would leave nested MappingProxyType values.
            serialized_decision = decision.to_dict()
            metadata = serialized_decision["metadata"]
            assert isinstance(metadata, dict)
            if agent_settings is not None:
                setting = agent_settings[decision.move.player.value]
                metadata.setdefault("guidance_mode", setting["guidance_mode"])
            frames.append(
                {
                    "move": move_number,
                    "state": state.to_dict(),
                    "last_move": {
                        "player": decision.move.player.value,
                        "coordinate": decision.move.coordinate.to_dict(),
                    },
                    "decision": {
                        "selected_move": {
                            "player": decision.move.player.value,
                            "coordinate": decision.move.coordinate.to_dict(),
                        },
                        "seed": decision.seed,
                        "metadata": metadata,
                    },
                }
            )
        move_count = len(match.moves)
        capacity = self._board_capacity(match.config.board)
        is_draw = match.game_result.value == "draw"
        first_metadata = (
            match.decisions[0].to_dict()["metadata"] if match.decisions else {}
        )
        assert isinstance(first_metadata, dict)
        return {
            "format": "twixt-ai-viewer-replay",
            "version": 1,
            "source": dict(source),
            "board": match.config.board.to_dict(),
            "agents": (
                {key: dict(value) for key, value in agent_settings.items()}
                if agent_settings is not None
                else match.config.to_dict()["agents"]
            ),
            "search": {
                "simulations": first_metadata.get("simulations"),
                "rollout_limit": first_metadata.get("rollout_limit"),
            },
            "frames": frames,
            "result": {
                "status": match.game_result.value,
                "winner": match.winner.value if match.winner is not None else None,
                "draw": is_draw,
                "move_count": move_count,
                "board_capacity": capacity,
                "filled_to_limit": is_draw and move_count == capacity,
            },
        }

    def generate(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "red",
            "black",
            "seed",
        }:
            raise ValueError("game request must contain exactly black, red, and seed")
        seed = payload["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        settings: dict[str, dict[str, object]] = {}
        agents: dict[str, Agent] = {}
        for side in (Player.RED.value, Player.BLACK.value):
            value = payload[side]
            if not isinstance(value, Mapping) or set(value) != {"mode", "checkpoint"}:
                raise ValueError(f"{side} must contain exactly checkpoint and mode")
            mode = value["mode"]
            if not isinstance(mode, str):
                raise ValueError(f"{side} mode must be a string")
            checkpoint = value["checkpoint"]
            agents[side] = self._agent(mode, checkpoint)
            settings[side] = {
                "mode": mode,
                "guidance_mode": (
                    "none" if mode == "non-neural-mcts" else mode.removeprefix("learned-")
                ),
                "checkpoint": checkpoint if mode != "non-neural-mcts" else None,
            }
        match = run_match(
            agents[Player.RED.value],
            agents[Player.BLACK.value],
            config=MatchConfig(
                board=BoardDimensions(10, 10),
                seed=seed,
                red_agent=str(settings[Player.RED.value]["mode"]),
                black_agent=str(settings[Player.BLACK.value]["mode"]),
            ),
        )
        return self._replay(
            match,
            source={"type": "generated", "seed": seed},
            agent_settings=settings,
        )

    def load_artifact(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, Mapping) or set(payload) != {"artifact"}:
            raise ValueError("artifact request must contain exactly artifact")
        artifact_id = payload["artifact"]
        if not isinstance(artifact_id, str):
            raise ValueError("artifact must be a string")
        path = self._artifact_map().get(artifact_id)
        if path is None:
            raise ValueError("unknown artifact")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("artifact must contain a JSON object")
        if value.get("format") == "twixt-ai-match":
            match = MatchResult.from_dict(value)
        elif value.get("format") == "twixt-ai-game-record":
            record = GameRecord.from_dict(value)
            # Bare records have no decision inspection; project an empty set.
            from twixt_ai.evaluation.match import MatchDecision

            decisions = tuple(MatchDecision(move, None, {}) for move in record.moves)
            match = MatchResult(
                MatchConfig(
                    board=record.initial_state.board,
                    red_agent="artifact-red",
                    black_agent="artifact-black",
                ),
                record,
                decisions,
            )
        else:
            raise ValueError("artifact has no replayable state history")
        return self._replay(match, source={"type": "artifact", "path": artifact_id})


__all__ = [
    "DEFAULT_VIEWER_SIMULATIONS",
    "MAX_LISTED_ARTIFACTS",
    "VIEWER_AGENT_MODES",
    "ViewerService",
]
