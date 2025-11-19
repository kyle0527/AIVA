module aiva/scan/go_scanners

go 1.21

require (
	github.com/rabbitmq/amqp091-go v1.10.0
	go.uber.org/zap v1.26.0
)

require go.uber.org/multierr v1.10.0 // indirect

// 本地路徑依賴 - 依照 aiva_common README 規範
replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../../features/common/go/aiva_common_go
