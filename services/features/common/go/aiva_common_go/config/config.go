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
	rabbitmqURL := getEnv("AIVA_RABBITMQ_URL", "")
	if rabbitmqURL == "" {
		// 若未設定完整 URL，嘗試組合式配置
		host := getEnv("AIVA_RABBITMQ_HOST", "localhost")
		port := getEnv("AIVA_RABBITMQ_PORT", "5672")
		user := getEnv("AIVA_RABBITMQ_USER", "")
		password := getEnv("AIVA_RABBITMQ_PASSWORD", "")
		vhost := getEnv("AIVA_RABBITMQ_VHOST", "/")

		if user != "" && password != "" {
			rabbitmqURL = fmt.Sprintf("amqp://%s:%s@%s:%s%s", user, password, host, port, vhost)
		} else {
			return nil, fmt.Errorf("AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set")
		}
	}

	config := &Config{
		RabbitMQURL: rabbitmqURL,
		ServiceName: serviceName,
		LogLevel:    getEnv("AIVA_LOG_LEVEL", "info"),
		TaskQueue:   getEnv("AIVA_TASK_QUEUE", fmt.Sprintf("tasks.function.%s", serviceName)),
		ResultQueue: getEnv("AIVA_RESULT_QUEUE", "findings.new"),
		Environment: getEnv("AIVA_ENVIRONMENT", "development"),
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
