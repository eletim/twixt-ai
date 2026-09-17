#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=${TWIXT_PORT:-8000}
CHECKPOINT="$SCRIPT_DIR/models/frozen/gen11/best.pt"
SERVER_PID=""
SERVE_OWNED=0
SERVE_PORT=""

cleanup() {
    if (( SERVE_OWNED )); then
        tailscale serve --https="$SERVE_PORT" off >/dev/null 2>&1 || true
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Gen11 checkpoint is missing: $CHECKPOINT" >&2
    exit 1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "Error: TWIXT_PORT must be between 1 and 65535." >&2
    exit 1
fi
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 -c 'import twixt_ai.backend.server; import torch' >/dev/null 2>&1; then
    echo "Error: Python backend dependencies are unavailable. Run: python3 -m pip install -e \".[models]\"" >&2
    exit 1
fi
if [[ -z "${TWIXT_PORT:-}" ]]; then
    for candidate in 8000 8001 8002 8003; do
        if python3 -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$candidate" 2>/dev/null; then
            PORT=$candidate
            break
        fi
    done
fi
if ! python3 -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT" 2>/dev/null; then
    echo "Error: local port $PORT is already in use; set TWIXT_PORT to a free port." >&2
    exit 1
fi
LOCAL_URL="http://127.0.0.1:$PORT"

python3 -m twixt_ai.backend --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
ready=0
for _ in {1..50}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    if python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=0.2).close()' "$LOCAL_URL/api/session" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 0.1
done
if (( ! ready )); then
    echo "Error: Twixt backend failed to start on $LOCAL_URL (check the error above or whether the port is in use)." >&2
    wait "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "Local:     $LOCAL_URL/"
echo "Viewer:    $LOCAL_URL/viewer"

if command -v tailscale >/dev/null 2>&1; then
    status=$(tailscale status --json 2>/dev/null) || status=""
    dns=$(python3 -c 'import json,sys; s=json.load(sys.stdin); print(s.get("Self",{}).get("DNSName","").rstrip(".") if s.get("BackendState")=="Running" else "")' <<<"$status" 2>/dev/null) || dns=""
    if [[ -n "$dns" ]]; then
        serve_status=$(tailscale serve status --json 2>/dev/null) || serve_status=""
        for candidate in 8765 8766 8767; do
            port_state=$(python3 -c '
import json,socket,sys
port=sys.argv[1]
try: serve=json.loads(sys.argv[2])
except ValueError: serve={}
status=json.load(sys.stdin)
handlers=[v for k,w in serve.get("Web",{}).items() if k.endswith(":"+port) for v in w.get("Handlers",{}).values()]
free=not handlers and port not in serve.get("TCP",{})
for ip in status.get("TailscaleIPs",[])[:1]:
    try:
        with socket.socket() as sock: sock.bind((ip,int(port)))
    except OSError: free=False
print("free" if free else "occupied")
' "$candidate" "$serve_status" <<<"$status")
            if [[ "$port_state" == "free" ]]; then
                SERVE_PORT=$candidate
                break
            fi
        done
        if [[ -n "$SERVE_PORT" ]]; then
            if tailscale serve --bg --yes --https="$SERVE_PORT" "$LOCAL_URL"; then
                SERVE_OWNED=1
                echo "Tailscale: https://$dns:$SERVE_PORT/"
                echo "Viewer:    https://$dns:$SERVE_PORT/viewer"
            else
                echo "Warning: Tailscale Serve setup failed; localhost remains available." >&2
            fi
        else
            echo "Warning: Tailscale Serve ports 8765-8767 are occupied; existing settings were preserved." >&2
        fi
    else
        echo "Warning: Tailscale is not connected; localhost remains available." >&2
    fi
else
    echo "Warning: Tailscale is unavailable; localhost remains available." >&2
fi

wait "$SERVER_PID"
