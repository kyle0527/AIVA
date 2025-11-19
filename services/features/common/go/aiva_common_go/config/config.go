package config

import (
	"fmt"
	"os"
)

// Config 統一的配置結構
type Config struct {
	RabbitMQURL string
	ServiceName string
	LogLevel    string
	TaskQueue   string
	ResultQueue string
	Environment string
}

// LoadConfig 從環境變數載入配置
func LoadConfig(serviceName string) (*Config, error) {
	// 遵循 12-factor app 原則，移除硬編碼認證資訊
	// 簡化環境變數命名 (移除 AIVA_ 前綴)
	rabbitmqURL := getEnv("RABBITMQ_URL", "")
	if rabbitmqURL == "" {
		// 若未設定完整 URL，嘗試組合式配置
		host := getEnv("RABBITMQ_HOST", "localhost")
		port := getEnv("RABBITMQ_PORT", "5672")
		user := getEnv("RABBITMQ_USER", "")
		password := getEnv("RABBITMQ_PASSWORD", "")
		vhost := getEnv("RABBITMQ_VHOST", "/")

		if user != "" && password != "" {
			rabbitmqURL = fmt.Sprintf("amqp://%s:%s@%s:%s%s", user, password, host, port, vhost)
		} else {
			return nil, fmt.Errorf("RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set")
		}
	}

	config := &Config{
		RabbitMQURL: rabbitmqURL,
		ServiceName: serviceName,
		LogLevel:    getEnv("LOG_LEVEL", "info"),
		TaskQueue:   getEnv("TASK_QUEUE", fmt.Sprintf("tasks.function.%s", serviceName)),
		ResultQueue: getEnv("RESULT_QUEUE", "findings.new"),
		Environment: getEnv("ENVIRONMENT", "development"),
	}

	// 驗證必要配置
	if config.RabbitMQURL == "" {
		return nil, fmt.Errorf("RABBITMQ_URL 不能為空")
	}

	return config, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
