"""Integration tests for the checkout launcher and Tailscale setup."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


PROJECT_ROOT = Path(__file__).parents[2]
START_SCRIPT = PROJECT_ROOT / "start.sh"


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run_launcher(tmp_path: Path, tailscale: str) -> tuple[str, str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stopped = tmp_path / "server-stopped"
    serve_log = tmp_path / "tailscale-serve.log"
    _executable(
        bin_dir / "twixt-ai-web",
        f"""#!/usr/bin/env bash
trap 'touch {stopped!s}; exit 0' TERM INT
while true; do sleep 0.05; done
""",
    )
    _executable(bin_dir / "tailscale", tailscale)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["SERVE_LOG"] = str(serve_log)
    process = subprocess.Popen(
        [str(START_SCRIPT)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not serve_log.exists():
            time.sleep(0.02)
    finally:
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 143
    assert stopped.exists()
    return stdout, stderr, serve_log.read_text(encoding="utf-8") if serve_log.exists() else ""


def test_launcher_configures_one_tailnet_root_proxy_and_prints_routes(tmp_path: Path) -> None:
    stdout, stderr, serve_call = _run_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "status --json" ]]; then
    printf '%s\\n' '{"BackendState":"Running","Self":{"DNSName":"twixt.example.ts.net."}}'
    exit 0
fi
printf '%s\\n' "$*" > "$SERVE_LOG"
""",
    )

    assert stderr == ""
    assert "Local Twixt UI:     http://127.0.0.1:8000/" in stdout
    assert "Local AI viewer:    http://127.0.0.1:8000/viewer" in stdout
    assert "Tailnet Twixt UI:   https://twixt.example.ts.net/" in stdout
    assert "Tailnet AI viewer:  https://twixt.example.ts.net/viewer" in stdout
    assert serve_call == "serve --bg --yes --set-path=/ http://127.0.0.1:8000\n"
    assert "funnel" not in serve_call


def test_launcher_keeps_local_server_when_tailscale_is_logged_out(tmp_path: Path) -> None:
    stdout, stderr, serve_call = _run_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "status --json" ]]; then
    printf '%s\\n' '{"BackendState":"NeedsLogin"}'
    exit 1
fi
printf '%s\\n' "$*" > "$SERVE_LOG"
""",
    )

    assert "http://127.0.0.1:8000/" in stdout
    assert "http://127.0.0.1:8000/viewer" in stdout
    assert "Tailscale is not logged in and running" in stderr
    assert serve_call == ""
