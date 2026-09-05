"""Checks that the initial Python architecture remains headless."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


HEADLESS_PACKAGES = (
    "game",
    "agents",
    "search",
    "models",
    "selfplay",
    "training",
    "evaluation",
)


def test_headless_packages_import_without_backend_or_ui() -> None:
    for package in HEADLESS_PACKAGES:
        importlib.import_module(f"twixt_ai.{package}")


def test_game_package_has_no_inverted_dependencies() -> None:
    game_root = Path(__file__).parents[1] / "src" / "twixt_ai" / "game"
    forbidden = {"agents", "search", "models", "selfplay", "training", "evaluation", "backend", "ui"}

    for source in game_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = set(node.module.split("."))
                assert parts.isdisjoint(forbidden), f"{source} imports {node.module}"
