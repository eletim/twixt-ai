"""Winning-path detection for canonical Twixt positions."""

from __future__ import annotations

from collections import deque

from .state import Coordinate, GameState, Player


def _is_start_border(state: GameState, player: Player, coordinate: Coordinate) -> bool:
    if player is Player.RED:
        return coordinate.y == 0 and 0 < coordinate.x < state.board.width - 1
    return coordinate.x == 0 and 0 < coordinate.y < state.board.height - 1


def _is_target_border(state: GameState, player: Player, coordinate: Coordinate) -> bool:
    if player is Player.RED:
        return (
            coordinate.y == state.board.height - 1
            and 0 < coordinate.x < state.board.width - 1
        )
    return (
        coordinate.x == state.board.width - 1
        and 0 < coordinate.y < state.board.height - 1
    )


def winning_path(state: GameState, player: Player) -> tuple[Coordinate, ...] | None:
    """Return one connected path between *player*'s target borders, if any.

    Red connects the north and south borders; Black connects the west and east
    borders. Only links owned by the requested player provide connectivity.
    The breadth-first traversal is deterministic and linear in the position's
    pegs and links. Returning the path as well as detecting it gives callers a
    useful witness without requiring them to repeat the graph traversal.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(player, Player):
        raise TypeError("player must be a Player")

    owned_coordinates = {
        peg.coordinate for peg in state.pegs if peg.owner is player
    }
    starts = sorted(
        coordinate
        for coordinate in owned_coordinates
        if _is_start_border(state, player, coordinate)
    )
    if not starts:
        return None

    adjacency: dict[Coordinate, list[Coordinate]] = {
        coordinate: [] for coordinate in owned_coordinates
    }
    for link in state.links:
        if link.owner is player:
            adjacency[link.start].append(link.end)
            adjacency[link.end].append(link.start)
    for neighbors in adjacency.values():
        neighbors.sort()

    queue = deque(starts)
    predecessor: dict[Coordinate, Coordinate | None] = {
        coordinate: None for coordinate in starts
    }
    while queue:
        coordinate = queue.popleft()
        if _is_target_border(state, player, coordinate):
            path = [coordinate]
            previous = predecessor[coordinate]
            while previous is not None:
                path.append(previous)
                previous = predecessor[previous]
            path.reverse()
            return tuple(path)

        for neighbor in adjacency[coordinate]:
            if neighbor not in predecessor:
                predecessor[neighbor] = coordinate
                queue.append(neighbor)

    return None


def has_winning_path(state: GameState, player: Player) -> bool:
    """Return whether *player* connects their two target borders."""

    return winning_path(state, player) is not None


__all__ = ["has_winning_path", "winning_path"]
