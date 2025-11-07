#!/usr/bin/env bash
set -euo pipefail
# Regenerate multi-language contracts from SSoT YAML
python services/aiva_common/tools/schema_codegen_tool.py --generate-all
echo "[OK] Contracts regenerated"
