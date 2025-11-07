#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t aiva/ssrf_worker:latest -f "$ROOT/services/features/function_ssrf/Dockerfile" "$ROOT"
