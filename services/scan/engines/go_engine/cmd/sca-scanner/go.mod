module github.com/kyle0527/aiva/services/scan/engines/go_engine/cmd/sca-scanner

go 1.23.1

require (
	github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca v0.0.0
	go.uber.org/zap v1.26.0
)

require (
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
)

replace github.com/kyle0527/aiva/services/scan/engines/go_engine/pkg/models => ../../pkg/models

replace github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca => ../../internal/sca

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../../../../../features/common/go/aiva_common_go
