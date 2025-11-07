#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t aiva/authn_go_worker:latest -f "$ROOT/services/features/function_authn_go/Dockerfile" "$ROOT"
