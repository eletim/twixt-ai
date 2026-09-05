"""Distribution-level smoke tests for the browser executable's assets."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).parents[2]


def test_installed_wheel_serves_all_ui_assets(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    install_dir = tmp_path / "installed"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(PROJECT_ROOT),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("twixt_ai-*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    smoke_test = f"""
import sys
from io import BytesIO

sys.path.insert(0, {str(install_dir)!r})
from twixt_ai.backend import GameApplication

application = GameApplication()
for path, marker in (
    ("/", b"<!doctype html>"),
    ("/app.js", b"const SVG_NS"),
    ("/styles.css", b":root"),
):
    response = {{}}
    def start_response(status, headers):
        response["status"] = status
        response["headers"] = dict(headers)
    body = b"".join(application({{
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path,
        "CONTENT_LENGTH": "0",
        "wsgi.input": BytesIO(),
    }}, start_response))
    assert response["status"] == "200 OK", (path, response, body)
    assert marker in body, (path, body)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-S", "-c", smoke_test],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
