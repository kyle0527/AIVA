module github.com/kyle0527/aiva/services/function/function_ssrf_go

go 1.21

require (
	github.com/rabbitmq/amqp091-go v1.10.0
	go.uber.org/zap v1.26.0
)

require (
	github.com/stretchr/testify v1.8.4 // indirect
	go.uber.org/multierr v1.11.0 // indirect
)

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../common/go/aiva_common_go
