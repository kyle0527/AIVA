#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/.."
CRYPTO_DIR="${ROOT_DIR}/services/features/function_crypto"
pushd "${CRYPTO_DIR}/rust_core" >/dev/null
# Build Python extension using maturin (ensure rust/cargo + Python are installed)
if ! command -v maturin >/dev/null; then
  pip install maturin
fi
maturin build --release --strip -i python3
# Install the built wheel into the system/venv
WHEEL=$(ls target/wheels/crypto_engine-*.whl | head -n1)
pip install "$WHEEL"
popd >/dev/null
echo "[OK] crypto_engine installed"
