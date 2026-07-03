#!/usr/bin/env bash
set -euo pipefail

# Run LIBERO in headless Docker mode on remote servers.
# Allows overriding args via env vars:
#   SERVER_ARGS (default: --env LIBERO)
#   CLIENT_ARGS (optional)
#   MUJOCO_GL  (default: egl)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SERVER_ARGS_VALUE="${SERVER_ARGS:---env LIBERO}"
MUJOCO_GL_VALUE="${MUJOCO_GL:-egl}"

if [[ "${1:-}" == "--quick" ]]; then
  CLIENT_ARGS_VALUE="${CLIENT_ARGS:---args.num-trials-per-task 2}"
  echo "Running quick check with CLIENT_ARGS=${CLIENT_ARGS_VALUE}"
else
  CLIENT_ARGS_VALUE="${CLIENT_ARGS:-}"
fi

if [[ -n "$CLIENT_ARGS_VALUE" ]]; then
  sudo env \
    SERVER_ARGS="$SERVER_ARGS_VALUE" \
    CLIENT_ARGS="$CLIENT_ARGS_VALUE" \
    MUJOCO_GL="$MUJOCO_GL_VALUE" \
    docker compose \
      -f examples/libero/compose.yml \
      -f examples/libero/compose.headless.yml \
      up --build
else
  sudo env \
    SERVER_ARGS="$SERVER_ARGS_VALUE" \
    MUJOCO_GL="$MUJOCO_GL_VALUE" \
    docker compose \
      -f examples/libero/compose.yml \
      -f examples/libero/compose.headless.yml \
      up --build
fi

