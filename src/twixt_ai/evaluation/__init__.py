"""Headless matches, tournaments, metrics, and model promotion."""

from .benchmark import (
    BENCHMARK_FORMAT,
    BENCHMARK_FORMAT_VERSION,
    AgentConfig,
    AgentFactory,
    BenchmarkConfig,
    BenchmarkGame,
    BenchmarkResult,
    run_benchmark,
)

from .match import (
    MATCH_FORMAT,
    MATCH_FORMAT_VERSION,
    MatchConfig,
    MatchDecision,
    MatchResult,
    run_match,
)

__all__ = [
    "BENCHMARK_FORMAT",
    "BENCHMARK_FORMAT_VERSION",
    "MATCH_FORMAT",
    "MATCH_FORMAT_VERSION",
    "AgentConfig",
    "AgentFactory",
    "BenchmarkConfig",
    "BenchmarkGame",
    "BenchmarkResult",
    "MatchConfig",
    "MatchDecision",
    "MatchResult",
    "run_benchmark",
    "run_match",
]
