"""Tests for the Mini dataset experiment command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.selfplay import experiment_cli


def test_cli_fails_when_smoke_stage_records_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "stages": {
            "smoke": {
                "selfplay": {"summary": {"aggregate": {"failed": 1}}}
            },
            "baseline": {
                "selfplay": {"summary": {"aggregate": {"failed": 0}}}
            },
        }
    }
    monkeypatch.setattr(
        experiment_cli, "run_mini_dataset_experiment", lambda output: report
    )

    assert experiment_cli.main(["--output-dir", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out) == report
