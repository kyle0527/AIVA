module github.com/kyle0527/aiva/services/scan/go_engine/cmd/ssrf-scanner

go 1.23.1

require (
	github.com/kyle0527/aiva/services/scan/go_engine/internal/ssrf v0.0.0
	github.com/kyle0527/aiva/services/scan/go_engine/pkg/models v0.0.0
)

replace github.com/kyle0527/aiva/services/scan/go_engine/internal/ssrf => ../../internal/ssrf
replace github.com/kyle0527/aiva/services/scan/go_engine/pkg/models => ../../pkg/models
