#!/usr/bin/env bash
set -euo pipefail
export AIVA_AMQP_URL=${AIVA_AMQP_URL:-"amqp://guest:guest@localhost:5672/"}
python -m services.features.function_postex
