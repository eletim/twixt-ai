"""Headless matches, tournaments, metrics, and model promotion."""

from .match import (
    MATCH_FORMAT,
    MATCH_FORMAT_VERSION,
    MatchConfig,
    MatchDecision,
    MatchResult,
    run_match,
)

__all__ = [
    "MATCH_FORMAT",
    "MATCH_FORMAT_VERSION",
    "MatchConfig",
    "MatchDecision",
    "MatchResult",
    "run_match",
]
