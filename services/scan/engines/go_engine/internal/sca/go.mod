module github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca

go 1.23.1

require (
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0
	github.com/sirupsen/logrus v1.9.3
	go.uber.org/zap v1.26.0
)

require (
	github.com/stretchr/testify v1.8.4 // indirect
	go.uber.org/goleak v1.3.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8 // indirect
)

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../../../../../features/common/go/aiva_common_go
