#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t aiva/ssrf_go:latest -f "$ROOT/services/scan/go_scanners/ssrf_scanner/Dockerfile" "$ROOT"
docker build -t aiva/cspm_go:latest -f "$ROOT/services/scan/go_scanners/cspm_scanner/Dockerfile" "$ROOT"
docker build -t aiva/sca_go:latest  -f "$ROOT/services/scan/go_scanners/sca_scanner/Dockerfile"  "$ROOT"
