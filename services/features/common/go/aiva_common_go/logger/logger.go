package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/buffer"
	"go.uber.org/zap/zapcore"
)

// AIVALogEntry 統一日誌結構
type AIVALogEntry struct {
	Timestamp string                 `json:"timestamp"`
	Level     string                 `json:"level"`
	Service   string                 `json:"service"`
	Module    string                 `json:"module"`
	Message   string                 `json:"message"`
	Logger    string                 `json:"logger"`
	TaskID    string                 `json:"task_id,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	UserID    string                 `json:"user_id,omitempty"`
	Trace     *TraceInfo             `json:"trace,omitempty"`
	Extra     map[string]interface{} `json:"extra,omitempty"`
	AI        *AIInfo                `json:"ai,omitempty"`
}

// TraceInfo 追蹤信息
type TraceInfo struct {
	File      string `json:"file"`
	Line      int    `json:"line"`
	Function  string `json:"function"`
	Goroutine int64  `json:"goroutine"`
}

// AIInfo AI 相關信息
type AIInfo struct {
	Confidence   float64 `json:"confidence,omitempty"`
	ModelVersion string  `json:"model_version,omitempty"`
	Prediction   string  `json:"prediction,omitempty"`
	Accuracy     float64 `json:"accuracy,omitempty"`
}

// CrossLanguageCall 跨語言調用信息
type CrossLanguageCall struct {
	Source          string  `json:"source"`
	Target          string  `json:"target"`
	Function        string  `json:"function"`
	ParametersHash  string  `json:"parameters_hash"`
	HasResult       bool    `json:"has_result"`
	HasError        bool    `json:"has_error"`
	ExecutionTimeMS float64 `json:"execution_time_ms,omitempty"`
}

// AIVAEncoder 自定義編碼器，輸出統一 JSON 格式
type AIVAEncoder struct {
	zapcore.Encoder
	serviceName string
	moduleName  string
}

func (enc *AIVAEncoder) Clone() zapcore.Encoder {
	return &AIVAEncoder{
		Encoder:     enc.Encoder.Clone(),
		serviceName: enc.serviceName,
		moduleName:  enc.moduleName,
	}
}

func (enc *AIVAEncoder) EncodeEntry(entry zapcore.Entry, fields []zapcore.Field) (*buffer.Buffer, error) {
	// 獲取調用者信息
	_, file, line, _ := runtime.Caller(7) // 調整調用深度

	// 獲取函數信息
	pc, _, _, _ := runtime.Caller(7)
	funcName := runtime.FuncForPC(pc).Name()

	logEntry := AIVALogEntry{
		Timestamp: entry.Time.Format(time.RFC3339),
		Level:     entry.Level.String(),
		Service:   enc.serviceName,
		Module:    enc.moduleName,
		Message:   entry.Message,
		Logger:    entry.LoggerName,
		Trace: &TraceInfo{
			File:     file,
			Line:     line,
			Function: funcName,
		},
		Extra: make(map[string]interface{}),
	}

	// 處理自定義字段
	for _, field := range fields {
		switch field.Key {
		case "task_id":
			logEntry.TaskID = field.String
		case "session_id":
			logEntry.SessionID = field.String
		case "user_id":
			logEntry.UserID = field.String
		case "confidence":
			if logEntry.AI == nil {
				logEntry.AI = &AIInfo{}
			}
			if f, ok := field.Interface.(float64); ok {
				logEntry.AI.Confidence = f
			}
		case "model_version":
			if logEntry.AI == nil {
				logEntry.AI = &AIInfo{}
			}
			logEntry.AI.ModelVersion = field.String
		case "prediction":
			if logEntry.AI == nil {
				logEntry.AI = &AIInfo{}
			}
			logEntry.AI.Prediction = field.String
		case "cross_language_call":
			logEntry.Extra["cross_language_call"] = field.Interface
		default:
			logEntry.Extra[field.Key] = field.Interface
		}
	}

	// 如果沒有額外字段，移除 extra
	if len(logEntry.Extra) == 0 {
		logEntry.Extra = nil
	}

	// 序列化為 JSON
	jsonBytes, err := json.Marshal(logEntry)
	if err != nil {
		return nil, err
	}

	buf := &buffer.Buffer{}
	buf.Write(jsonBytes)
	buf.AppendByte('\n')
	return buf, nil
}

// NewLogger 建立標準化的統一格式日誌實例
func NewLogger(serviceName, moduleName string) (*zap.Logger, error) {
	// 創建自定義編碼器配置
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.CapitalLevelEncoder,
		EncodeTime:     zapcore.RFC3339TimeEncoder,
		EncodeDuration: zapcore.SecondsDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// 創建自定義編碼器
	encoder := &AIVAEncoder{
		Encoder:     zapcore.NewJSONEncoder(encoderConfig),
		serviceName: serviceName,
		moduleName:  moduleName,
	}

	// 創建核心
	core := zapcore.NewCore(
		encoder,
		zapcore.AddSync(os.Stdout),
		zap.InfoLevel,
	)

	// 創建日誌器
	logger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))

	return logger, nil
}

// NewDevelopmentLogger 建立開發環境日誌(保持兼容性)
func NewDevelopmentLogger(serviceName string) (*zap.Logger, error) {
	return NewLogger(serviceName, "development")
}

// LogCrossLanguageCall 記錄跨語言調用
func LogCrossLanguageCall(
	logger *zap.Logger,
	source, target, function string,
	parametersHash string,
	hasResult, hasError bool,
	executionTimeMS float64,
	message string,
) {
	call := CrossLanguageCall{
		Source:          source,
		Target:          target,
		Function:        function,
		ParametersHash:  parametersHash,
		HasResult:       hasResult,
		HasError:        hasError,
		ExecutionTimeMS: executionTimeMS,
	}

	if hasError {
		logger.Error(message, zap.Any("cross_language_call", call))
	} else if hasResult {
		logger.Info(message, zap.Any("cross_language_call", call))
	} else {
		logger.Debug(message, zap.Any("cross_language_call", call))
	}
}

// LogAIDecision 記錄 AI 決策
func LogAIDecision(
	logger *zap.Logger,
	decision string,
	confidence float64,
	modelVersion string,
	taskID string,
) {
	logger.Info(fmt.Sprintf("AI Decision: %s", decision),
		zap.Float64("confidence", confidence),
		zap.String("model_version", modelVersion),
		zap.String("prediction", decision),
		zap.String("task_id", taskID),
	)
}
