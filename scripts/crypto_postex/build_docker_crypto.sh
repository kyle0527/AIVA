#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/.."
docker build -t aiva/crypto_worker:latest -f "${ROOT_DIR}/services/features/function_crypto/Dockerfile" "${ROOT_DIR}"
