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
