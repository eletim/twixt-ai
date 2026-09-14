#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_URL="http://127.0.0.1:8000"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

wait_for_server() {
    for _ in {1..50}; do
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        if python3 -c '
import socket
with socket.create_connection(("127.0.0.1", 8000), timeout=0.1):
    pass
' 2>/dev/null; then
            sleep 0.1
            kill -0 "$SERVER_PID" 2>/dev/null && return 0
        fi
        sleep 0.1
    done
    return 1
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v twixt-ai-web >/dev/null 2>&1; then
    echo "Error: twixt-ai-web is not installed. Run: python -m pip install -e ." >&2
    exit 127
fi

cd "$SCRIPT_DIR"
twixt-ai-web --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

if ! wait_for_server; then
    echo "Error: twixt-ai-web failed to start on $LOCAL_URL." >&2
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        exit 1
    fi
    wait "$SERVER_PID"
    SERVER_STATUS=$?
    SERVER_PID=""
    (( SERVER_STATUS == 0 )) && SERVER_STATUS=1
    exit "$SERVER_STATUS"
fi

echo "Local Twixt UI:     $LOCAL_URL/"
echo "Local AI viewer:    $LOCAL_URL/viewer"

if ! command -v tailscale >/dev/null 2>&1; then
    echo "Warning: Tailscale is unavailable; the Twixt UI is available locally only." >&2
else
    TAILSCALE_STATUS=$(tailscale status --json 2>/dev/null) || TAILSCALE_STATUS=""
    read -r TAILSCALE_STATE TAILSCALE_DNS_NAME < <(
        python3 -c '
import json, sys
try:
    status = json.load(sys.stdin)
except (json.JSONDecodeError, OSError):
    status = {}
print(status.get("BackendState", ""), status.get("Self", {}).get("DNSName", ""))
' <<<"$TAILSCALE_STATUS"
    )
    TAILSCALE_DNS_NAME=${TAILSCALE_DNS_NAME%.}

    if [[ "$TAILSCALE_STATE" != "Running" || -z "$TAILSCALE_DNS_NAME" ]]; then
        echo "Warning: Tailscale is not logged in and running; the Twixt UI is available locally only." >&2
    elif tailscale serve --bg --yes --set-path=/ "$LOCAL_URL"; then
        TAILNET_URL="https://$TAILSCALE_DNS_NAME"
        echo "Tailnet Twixt UI:   $TAILNET_URL/"
        echo "Tailnet AI viewer:  $TAILNET_URL/viewer"
    else
        echo "Warning: Tailscale Serve could not be configured; the Twixt UI is available locally only." >&2
    fi
fi

wait "$SERVER_PID"
