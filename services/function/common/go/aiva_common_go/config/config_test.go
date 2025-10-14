package config

import (
	"os"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	// 設置測試環境變數
	os.Setenv("RABBITMQ_URL", "amqp://test:test@localhost:5672/")
	os.Setenv("LOG_LEVEL", "debug")
	defer os.Unsetenv("RABBITMQ_URL")
	defer os.Unsetenv("LOG_LEVEL")

	cfg, err := LoadConfig("test_service")
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.ServiceName != "test_service" {
		t.Errorf("ServiceName = %v, want %v", cfg.ServiceName, "test_service")
	}

	if cfg.RabbitMQURL != "amqp://test:test@localhost:5672/" {
		t.Errorf("RabbitMQURL = %v, want %v", cfg.RabbitMQURL, "amqp://test:test@localhost:5672/")
	}

	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %v, want %v", cfg.LogLevel, "debug")
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	// 清除環境變數以測試預設值
	os.Unsetenv("RABBITMQ_URL")
	os.Unsetenv("LOG_LEVEL")

	cfg, err := LoadConfig("test_service")
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.LogLevel != "info" {
		t.Errorf("Default LogLevel = %v, want %v", cfg.LogLevel, "info")
	}

	if cfg.Environment != "development" {
		t.Errorf("Default Environment = %v, want %v", cfg.Environment, "development")
	}
}
