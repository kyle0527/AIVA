#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t aiva/idor_worker:latest -f "$ROOT/services/features/function_idor/Dockerfile" "$ROOT"
