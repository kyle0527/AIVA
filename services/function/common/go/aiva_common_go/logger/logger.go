package logger

import (
	"go.uber.org/zap"
)

// NewLogger 建立標準化的日誌實例
func NewLogger(serviceName string) (*zap.Logger, error) {
	config := zap.NewProductionConfig()

	// 自訂配置
	config.OutputPaths = []string{"stdout"}
	config.ErrorOutputPaths = []string{"stderr"}

	// 添加服務名稱到所有日誌
	config.InitialFields = map[string]interface{}{
		"service": serviceName,
	}

	logger, err := config.Build()
	if err != nil {
		return nil, err
	}

	return logger, nil
}

// NewDevelopmentLogger 建立開發環境日誌(更友善的輸出)
func NewDevelopmentLogger(serviceName string) (*zap.Logger, error) {
	config := zap.NewDevelopmentConfig()
	config.InitialFields = map[string]interface{}{
		"service": serviceName,
	}

	return config.Build()
}
