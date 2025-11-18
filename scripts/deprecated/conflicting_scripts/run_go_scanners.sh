#!/usr/bin/env bash
set -euo pipefail
docker run --rm -e AIVA_AMQP_URL="${AIVA_AMQP_URL:-amqp://guest:guest@localhost:5672/}" -e SCAN_TASKS_SSRF_GO="${SCAN_TASKS_SSRF_GO:-SCAN_TASKS_SSRF_GO}" -e SCAN_RESULTS_QUEUE="${SCAN_RESULTS_QUEUE:-SCAN_RESULTS}" aiva/ssrf_go:latest &
docker run --rm -e AIVA_AMQP_URL="${AIVA_AMQP_URL:-amqp://guest:guest@localhost:5672/}" -e SCAN_TASKS_CSPM_GO="${SCAN_TASKS_CSPM_GO:-SCAN_TASKS_CSPM_GO}" -e SCAN_RESULTS_QUEUE="${SCAN_RESULTS_QUEUE:-SCAN_RESULTS}" aiva/cspm_go:latest &
docker run --rm -e AIVA_AMQP_URL="${AIVA_AMQP_URL:-amqp://guest:guest@localhost:5672/}" -e SCAN_TASKS_SCA_GO="${SCAN_TASKS_SCA_GO:-SCAN_TASKS_SCA_GO}" -e SCAN_RESULTS_QUEUE="${SCAN_RESULTS_QUEUE:-SCAN_RESULTS}" aiva/sca_go:latest &
wait
