#!/usr/bin/env bash
set -euo pipefail
export AMQP_URL="${AMQP_URL:-${AIVA_AMQP_URL:-amqp://guest:guest@localhost:5672/}}"
/usr/local/bin/authn_worker || authn_worker
