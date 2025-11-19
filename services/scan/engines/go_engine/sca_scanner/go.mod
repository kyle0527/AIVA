module github.com/kyle0527/aiva/services/scan/go_scanners/sca_scanner

go 1.21

require (
	aiva/scan/go_scanners/common v0.0.0
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0
	github.com/rabbitmq/amqp091-go v1.10.0
	go.uber.org/zap v1.26.0
)

require (
	go.uber.org/multierr v1.11.0 // indirect
)

replace aiva/scan/go_scanners/common => ../common
replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../../../../features/common/go/aiva_common_go
