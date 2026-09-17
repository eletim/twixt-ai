"""Cheap, interpretable features for evaluating Twixt positions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, overload

from twixt_ai.game import Coordinate, GameState, Link, Player, links_cross


TERMINAL_SCORE = 1_000_000.0
"""Score assigned to a win before applying the requested perspective."""

_KNIGHT_OFFSETS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)


# Issue #158: 2 = endpoint, 1 = required empty cell, 0 = don't care.
_EFFECTIVE_CONNECTION_PATTERNS = (
    (
        (0, 0, 1, 0, 0, 0),
        (0, 2, 1, 1, 0, 0),
        (1, 1, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 1),
        (0, 1, 1, 1, 2, 0),
        (0, 0, 0, 1, 0, 0),
    ),
    (
        (0, 0, 0, 1, 0, 0),
        (0, 0, 1, 1, 2, 0),
        (0, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 0),
        (0, 2, 1, 1, 1, 0),
        (0, 0, 1, 0, 0, 0),
    ),
    (
        (0, 0, 0, 0, 0),
        (0, 1, 2, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 2, 1, 0),
        (0, 0, 0, 0, 0),
    ),
    (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 1, 1, 1, 1, 1, 0),
        (0, 2, 1, 1, 1, 2, 0),
        (0, 1, 1, 1, 1, 1, 0),
        (0, 0, 0, 0, 0, 0, 0),
    ),
)


def _pattern_offsets(
    pattern: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    first, second = (
        (x, y)
        for y, row in enumerate(pattern)
        for x, cell in enumerate(row)
        if cell == 2
    )
    empty = tuple(
        (x - first[0], y - first[1])
        for y, row in enumerate(pattern)
        for x, cell in enumerate(row)
        if cell == 1
    )
    return (second[0] - first[0], second[1] - first[1]), empty


_EFFECTIVE_CONNECTION_OFFSETS = tuple(
    _pattern_offsets(pattern) for pattern in _EFFECTIVE_CONNECTION_PATTERNS
)


def _effective_connections(
    state: GameState, owned: set[Coordinate]
) -> Iterable[tuple[Coordinate, Coordinate]]:
    """Match the four fixed patterns without constructing rule Links.

    Required empty cells must exist on the board; don't-care cells need not.
    """

    occupied = {(peg.coordinate.x, peg.coordinate.y) for peg in state.pegs}
    owned_points = {(point.x, point.y): point for point in owned}
    for start in owned:
        for (dx, dy), empty in _EFFECTIVE_CONNECTION_OFFSETS:
            end = owned_points.get((start.x + dx, start.y + dy))
            if end is None:
                continue
            if all(
                0 <= start.x + ex < state.board.width
                and 0 <= start.y + ey < state.board.height
                and (start.x + ex, start.y + ey) not in occupied
                for ex, ey in empty
            ):
                yield start, end


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Relative importance of each intentionally small feature family."""

    progress: float = 12.0
    connectivity: float = 3.0
    threats: float = 1.0
    blocking: float = 1.0

    def __post_init__(self) -> None:
        for name in ("progress", "connectivity", "threats", "blocking"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} weight must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} weight must be finite")


DEFAULT_WEIGHTS = HeuristicWeights()


@dataclass(frozen=True, slots=True)
class PositionFeatures:
    """Unweighted features measured for one player.

    ``progress`` is the best component's normalized goal-axis span using real
    links and the four fixed effective-connection patterns from Issue #158,
    with a bonus for touching either goal edge. ``connectivity`` rewards real links
    and pegs joined into non-singleton real-link components. ``threats`` counts open
    links that could be made by one placement; ``blocked`` counts such links
    prevented specifically by an opponent link. Opportunity counts are divided
    by eight to keep their scale comparable with the other features.
    """

    progress: float
    connectivity: float
    threats: float
    blocked: float


@dataclass(frozen=True, slots=True)
class EvaluationBreakdown:
    """A scalar evaluation together with its inspectable inputs and terms."""

    player: Player
    score: float
    player_features: PositionFeatures
    opponent_features: PositionFeatures
    contributions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contributions", MappingProxyType(dict(self.contributions))
        )


def _components(
    state: GameState, player: Player, *, effective: bool = False
) -> tuple[frozenset[Coordinate], ...]:
    owned = {peg.coordinate for peg in state.pegs if peg.owner is player}
    adjacency = {coordinate: set() for coordinate in owned}
    for link in state.links:
        if link.owner is player:
            adjacency[link.start].add(link.end)
            adjacency[link.end].add(link.start)

    if effective:
        for start, end in _effective_connections(state, owned):
            adjacency[start].add(end)
            adjacency[end].add(start)

    remaining = set(owned)
    components: list[frozenset[Coordinate]] = []
    while remaining:
        pending = [min(remaining)]
        component: set[Coordinate] = set()
        while pending:
            coordinate = pending.pop()
            if coordinate in component:
                continue
            component.add(coordinate)
            remaining.discard(coordinate)
            pending.extend(adjacency[coordinate] - component)
        components.append(frozenset(component))
    return tuple(components)


def _axis_value(player: Player, coordinate: Coordinate) -> int:
    return coordinate.y if player is Player.RED else coordinate.x


def _is_playable(state: GameState, player: Player, coordinate: Coordinate) -> bool:
    if not state.board.contains(coordinate):
        return False
    if player is Player.RED:
        return coordinate.x not in (0, state.board.width - 1)
    return coordinate.y not in (0, state.board.height - 1)


def _progress(
    state: GameState, player: Player, components: tuple[frozenset[Coordinate], ...]
) -> float:
    extent = state.board.height - 1 if player is Player.RED else state.board.width - 1
    if extent <= 0 or not components:
        return 0.0

    best = 0.0
    for component in components:
        positions = tuple(_axis_value(player, coordinate) for coordinate in component)
        span = (max(positions) - min(positions)) / extent
        border_bonus = 0.5 * (0 in positions) + 0.5 * (extent in positions)
        best = max(best, span + border_bonus)
    return best


LinkBuckets = Mapping[tuple[int, int], tuple[Link, ...]]


def _index_links(links: Iterable[Link]) -> LinkBuckets:
    """Group links into the one or two unit cells their interiors can cross."""

    buckets: dict[tuple[int, int], list[Link]] = {}
    for link in links:
        minimum_x = min(link.start.x, link.end.x)
        maximum_x = max(link.start.x, link.end.x)
        minimum_y = min(link.start.y, link.end.y)
        maximum_y = max(link.start.y, link.end.y)
        for x in range(minimum_x, maximum_x):
            for y in range(minimum_y, maximum_y):
                buckets.setdefault((x, y), []).append(link)
    return {key: tuple(values) for key, values in buckets.items()}


def _nearby_links(candidate: Link, link_buckets: LinkBuckets) -> set[Link]:
    """Return only links whose bounding boxes can overlap ``candidate``."""

    nearby: set[Link] = set()
    minimum_x = min(candidate.start.x, candidate.end.x)
    maximum_x = max(candidate.start.x, candidate.end.x)
    minimum_y = min(candidate.start.y, candidate.end.y)
    maximum_y = max(candidate.start.y, candidate.end.y)
    for x in range(minimum_x, maximum_x):
        for y in range(minimum_y, maximum_y):
            nearby.update(link_buckets.get((x, y), ()))
    return nearby


def _opportunities(
    state: GameState, player: Player, link_buckets: LinkBuckets
) -> tuple[float, float]:
    occupied = {peg.coordinate for peg in state.pegs}
    open_links = 0
    opponent_blocked = 0

    for peg in state.pegs:
        if peg.owner is not player:
            continue
        for dx, dy in _KNIGHT_OFFSETS:
            x = peg.coordinate.x + dx
            y = peg.coordinate.y + dy
            if x < 0 or y < 0:
                continue
            endpoint = Coordinate(x, y)
            if endpoint in occupied or not _is_playable(state, player, endpoint):
                continue
            candidate = Link(player, peg.coordinate, endpoint)
            has_crossing = False
            blocked_by_opponent = False
            for link in _nearby_links(candidate, link_buckets):
                if not links_cross(candidate, link):
                    continue
                has_crossing = True
                if link.owner is player.opponent:
                    blocked_by_opponent = True
                    break
            if not has_crossing:
                open_links += 1
            elif blocked_by_opponent:
                opponent_blocked += 1

    return open_links / 8.0, opponent_blocked / 8.0


def _position_features(
    state: GameState, player: Player, link_buckets: LinkBuckets
) -> PositionFeatures:
    components = _components(state, player)
    links = sum(link.owner is player for link in state.links)
    joined_pegs = sum(max(0, len(component) - 1) for component in components)
    threats, blocked = _opportunities(state, player, link_buckets)
    return PositionFeatures(
        progress=_progress(state, player, _components(state, player, effective=True)),
        connectivity=float(links + joined_pegs),
        threats=threats,
        blocked=blocked,
    )


def position_features(state: GameState, player: Player) -> PositionFeatures:
    """Return deterministic, unweighted heuristic features for ``player``."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(player, Player):
        raise TypeError("player must be a Player")
    return _position_features(state, player, _index_links(state.links))


@overload
def evaluate_position(
    state: GameState,
    player: Player,
    *,
    debug: Literal[False] = False,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> float: ...


@overload
def evaluate_position(
    state: GameState,
    player: Player,
    *,
    debug: Literal[True],
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> EvaluationBreakdown: ...


def evaluate_position(
    state: GameState,
    player: Player,
    *,
    debug: bool = False,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> float | EvaluationBreakdown:
    """Evaluate ``state`` as a stable scalar from ``player``'s perspective.

    Non-terminal scores are a weighted difference between the requested
    player's features and their opponent's. Set ``debug=True`` to receive the
    same score with every feature contribution exposed. Wins and losses use a
    fixed score well outside the range of ordinary board features.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(player, Player):
        raise TypeError("player must be a Player")
    if not isinstance(debug, bool):
        raise TypeError("debug must be a bool")
    if not isinstance(weights, HeuristicWeights):
        raise TypeError("weights must be HeuristicWeights")

    link_buckets = _index_links(state.links)
    own = _position_features(state, player, link_buckets)
    opponent = _position_features(state, player.opponent, link_buckets)
    if state.is_terminal:
        winner = state.winner
        terminal = 0.0
        if winner is player:
            terminal = TERMINAL_SCORE
        elif winner is player.opponent:
            terminal = -TERMINAL_SCORE
        contributions = {
            "progress": 0.0,
            "connectivity": 0.0,
            "threats": 0.0,
            "blocking": 0.0,
            "terminal": terminal,
        }
    else:
        contributions = {
            "progress": weights.progress * (own.progress - opponent.progress),
            "connectivity": weights.connectivity
            * (own.connectivity - opponent.connectivity),
            "threats": weights.threats * (own.threats - opponent.threats),
            # Being blocked is bad; blocking the opponent is good.
            "blocking": weights.blocking * (opponent.blocked - own.blocked),
            "terminal": 0.0,
        }
    score = sum(contributions.values())
    if not debug:
        return score
    return EvaluationBreakdown(player, score, own, opponent, contributions)


__all__ = [
    "DEFAULT_WEIGHTS",
    "TERMINAL_SCORE",
    "EvaluationBreakdown",
    "HeuristicWeights",
    "PositionFeatures",
    "evaluate_position",
    "position_features",
]
