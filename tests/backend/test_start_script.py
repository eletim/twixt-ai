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


def _run_launcher(
    tmp_path: Path, tailscale: str, server: str | None = None
) -> tuple[str, str, str, int, bool]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stopped = tmp_path / "server-stopped"
    serve_log = tmp_path / "tailscale-serve.log"
    if server is None:
        server = f"""#!/usr/bin/env bash
cleanup() {{
    kill "$HTTP_PID" 2>/dev/null || true
    wait "$HTTP_PID" 2>/dev/null || true
    touch {stopped!s}
    exit 0
}}
trap cleanup TERM INT
python3 -m http.server 8000 --bind 127.0.0.1 >/dev/null 2>&1 &
HTTP_PID=$!
wait "$HTTP_PID"
"""
    _executable(bin_dir / "twixt-ai-web", server)
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
        while (
            time.monotonic() < deadline
            and not serve_log.exists()
            and process.poll() is None
        ):
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)

    return (
        stdout,
        stderr,
        serve_log.read_text(encoding="utf-8") if serve_log.exists() else "",
        process.returncode,
        stopped.exists(),
    )


def test_launcher_configures_one_tailnet_root_proxy_and_prints_routes(tmp_path: Path) -> None:
    stdout, stderr, serve_call, returncode, stopped = _run_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "status --json" ]]; then
    printf '%s\\n' '{"BackendState":"Running","Self":{"DNSName":"twixt.example.ts.net."}}'
    exit 0
fi
printf '%s\\n' "$*" > "$SERVE_LOG"
""",
    )

    assert returncode == 143
    assert stopped
    assert stderr == ""
    assert "Local Twixt UI:     http://127.0.0.1:8000/" in stdout
    assert "Local AI viewer:    http://127.0.0.1:8000/viewer" in stdout
    assert "Tailnet Twixt UI:   https://twixt.example.ts.net/" in stdout
    assert "Tailnet AI viewer:  https://twixt.example.ts.net/viewer" in stdout
    assert serve_call == "serve --bg --yes --set-path=/ http://127.0.0.1:8000\n"
    assert "funnel" not in serve_call


def test_launcher_keeps_local_server_when_tailscale_is_logged_out(tmp_path: Path) -> None:
    stdout, stderr, serve_call, returncode, stopped = _run_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "status --json" ]]; then
    printf '%s\\n' '{"BackendState":"NeedsLogin"}'
    exit 1
fi
printf '%s\\n' "$*" > "$SERVE_LOG"
""",
    )

    assert returncode == 143
    assert stopped
    assert "http://127.0.0.1:8000/" in stdout
    assert "http://127.0.0.1:8000/viewer" in stdout
    assert "Tailscale is not logged in and running" in stderr
    assert serve_call == ""


def test_launcher_does_not_configure_serve_when_server_fails(tmp_path: Path) -> None:
    stdout, stderr, serve_call, returncode, stopped = _run_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "status --json" ]]; then
    printf '%s\\n' '{"BackendState":"Running","Self":{"DNSName":"twixt.example.ts.net."}}'
    exit 0
fi
printf '%s\\n' "$*" > "$SERVE_LOG"
""",
        server="""#!/usr/bin/env bash
echo "address already in use" >&2
exit 1
""",
    )

    assert returncode == 1
    assert not stopped
    assert stdout == ""
    assert "address already in use" in stderr
    assert "twixt-ai-web failed to start" in stderr
    assert serve_call == ""
