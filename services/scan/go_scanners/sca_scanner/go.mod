module github.com/kyle0527/aiva/services/scan/go_scanners/sca_scanner

go 1.21

require (
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0-00010101000000-000000000000
	go.uber.org/zap v1.26.0
)

require (
	github.com/rabbitmq/amqp091-go v1.10.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
)

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../../../features/common/go/aiva_common_go
