"""Checkout launcher smoke tests with an isolated Tailscale command."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch(tmp_path: Path, *, connected: bool, occupied: bool = False):
    port = free_port()
    fake = tmp_path / "tailscale"
    fake.write_text("""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['TAILSCALE_LOG'], 'a') as log: log.write(' '.join(args) + '\\n')
if args == ['status', '--json']:
    print(json.dumps({'BackendState': 'Running' if os.environ['CONNECTED'] == '1' else 'NeedsLogin', 'Self': {'DNSName': 'test.example.ts.net.'}}))
elif args == ['serve', 'status', '--json']:
    print(json.dumps({'TCP': {'8765': {'HTTPS': True}}} if os.environ['OCCUPIED'] == '1' else {}))
""", encoding="utf-8")
    fake.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f"{tmp_path}:{env['PATH']}", TWIXT_PORT=str(port),
               TAILSCALE_LOG=str(tmp_path / "calls"), CONNECTED=str(int(connected)),
               OCCUPIED=str(int(occupied)))
    process = subprocess.Popen([str(ROOT / "start.sh")], cwd=tmp_path, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/session", timeout=.2) as response:
                assert json.load(response)["gen11_available"] is True
            break
        except OSError:
            time.sleep(.1)
    else:
        process.terminate()
        out, err = process.communicate(timeout=5)
        raise AssertionError(f"launcher did not become ready: {out} {err}")
    # The server becomes reachable just before the launcher prints addresses
    # and configures Serve; let that short setup phase finish.
    time.sleep(.6)
    return process, port, tmp_path / "calls"


def stop(process: subprocess.Popen[str]):
    process.send_signal(signal.SIGTERM)
    return process.communicate(timeout=10)


def test_local_launcher_without_tailscale_connection(tmp_path: Path) -> None:
    process, port, calls = launch(tmp_path, connected=False)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/viewer") as response:
            assert b'id="board"' in response.read()
    finally:
        out, err = stop(process)
    assert process.returncode == 143
    assert f"http://127.0.0.1:{port}/" in out
    assert "not connected" in err
    assert "serve --bg" not in calls.read_text()


def test_serve_is_scoped_and_cleaned_up(tmp_path: Path) -> None:
    process, port, calls = launch(tmp_path, connected=True)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            assert b'Human vs AI' in response.read()
    finally:
        out, err = stop(process)
    commands = calls.read_text()
    assert process.returncode == 143
    assert "https://test.example.ts.net:8765/" in out
    assert "serve --bg --yes --https=8765" in commands
    assert "serve --https=8765 off" in commands
    assert "funnel" not in commands
    assert "Warning:" not in err


def test_existing_serve_port_is_preserved(tmp_path: Path) -> None:
    process, _, calls = launch(tmp_path, connected=True, occupied=True)
    out, err = stop(process)
    assert "https://test.example.ts.net:8766/" in out
    assert "serve --bg --yes --https=8766" in calls.read_text()
    assert "serve --https=8766 off" in calls.read_text()
    assert "serve --https=8765 off" not in calls.read_text()


def test_repeated_start_reports_port_conflict(tmp_path: Path) -> None:
    process, port, calls = launch(tmp_path, connected=False)
    try:
        env = os.environ.copy()
        env.update(PATH=f"{tmp_path}:{env['PATH']}", TWIXT_PORT=str(port),
                   TAILSCALE_LOG=str(calls), CONNECTED="0", OCCUPIED="0")
        repeat = subprocess.run([str(ROOT / "start.sh")], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=10)
        assert repeat.returncode == 1
        assert f"local port {port} is already in use" in repeat.stderr
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/viewer") as response:
            assert response.status == 200
    finally:
        stop(process)
