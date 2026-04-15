#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd python

echo "Testing API at ${BASE_URL}"

# 1) Create a session
create_resp="$(curl -sS -X POST "${BASE_URL}/sessions" -H 'Content-Type: application/json')"
session_id="$(printf '%s' "$create_resp" | python -c 'import json,sys; print(json.load(sys.stdin).get("session_id",""))')"

if [[ -z "$session_id" ]]; then
  echo "FAIL: could not parse session_id from create response: $create_resp" >&2
  exit 1
fi

echo "PASS: created session ${session_id}"

# 2) Send a chat message
chat_status="$(curl -sS -o /tmp/chat_resp.json -w '%{http_code}' \
  -X POST "${BASE_URL}/sessions/${session_id}/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello from curl test"}')"

if [[ "$chat_status" != "200" ]]; then
  echo "FAIL: chat endpoint returned status ${chat_status}" >&2
  cat /tmp/chat_resp.json >&2
  exit 1
fi

reply="$(python -c 'import json,sys; print(json.load(open("/tmp/chat_resp.json")).get("reply",""))')"
if [[ -z "$reply" ]]; then
  echo "FAIL: empty reply from chat endpoint" >&2
  cat /tmp/chat_resp.json >&2
  exit 1
fi

echo "PASS: chat reply received"

# 3) Get history and verify it contains system + user + assistant
history_status="$(curl -sS -o /tmp/history_resp.json -w '%{http_code}' \
  "${BASE_URL}/sessions/${session_id}/history")"

if [[ "$history_status" != "200" ]]; then
  echo "FAIL: history endpoint returned status ${history_status}" >&2
  cat /tmp/history_resp.json >&2
  exit 1
fi

history_count="$(python -c 'import json; print(len(json.load(open("/tmp/history_resp.json")).get("messages", [])))')"
if [[ "$history_count" -lt 3 ]]; then
  echo "FAIL: expected at least 3 history messages, got ${history_count}" >&2
  cat /tmp/history_resp.json >&2
  exit 1
fi

echo "PASS: history contains ${history_count} messages"

# 4) Delete session and ensure it is gone
delete_status="$(curl -sS -o /tmp/delete_resp.txt -w '%{http_code}' \
  -X DELETE "${BASE_URL}/sessions/${session_id}")"

if [[ "$delete_status" != "204" ]]; then
  echo "FAIL: delete endpoint returned status ${delete_status}" >&2
  cat /tmp/delete_resp.txt >&2
  exit 1
fi

after_delete_status="$(curl -sS -o /tmp/history_after_delete.json -w '%{http_code}' \
  "${BASE_URL}/sessions/${session_id}/history")"

if [[ "$after_delete_status" != "404" ]]; then
  echo "FAIL: expected 404 after delete, got ${after_delete_status}" >&2
  cat /tmp/history_after_delete.json >&2
  exit 1
fi

echo "PASS: deleted session returns 404 on history"
echo "All API smoke tests passed."
