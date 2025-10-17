# AIVA 多語言架構規劃方案

## 現況分析

### 當前語言分布
- **Python**: 主要服務（Core, Integration, 大部分 Function 模組）
- **Node.js/TypeScript**: 動態掃描（aiva_scan_node）
- **Rust**: 部分 Function 模組（function_sast_rust, info_gatherer_rust）
- **Go**: 部分 Function 模組（function_authn_go, function_cspm_go, function_ssrf_go）

### 各語言特性對比

| 語言 | 性能 | 記憶體效率 | 開發速度 | 生態系統 | 部署複雜度 | 適用場景 |
|------|------|------------|----------|----------|------------|----------|
| **Python** | 中等 | 低 | 高 | 豐富 | 低 | 快速原型、AI/ML、業務邏輯 |
| **Rust** | 極高 | 極高 | 低 | 中等 | 中等 | 系統級、安全關鍵、高性能計算 |
| **Go** | 高 | 高 | 中等 | 良好 | 低 | 併發服務、網路程式、微服務 |
| **Node.js** | 中高 | 中等 | 高 | 豐富 | 低 | I/O密集、即時應用、前端整合 |

## 建議的語言職責分配

### 🔥 **高性能/安全關鍵 - Rust**

**適用模組**:
- `function_sast_rust` ✅ (已實現)
- `info_gatherer_rust` ✅ (已實現)
- **新增建議**:
  - `function_sqli` (SQL 注入檢測需要高性能解析)
  - `function_xss` (XSS 檢測需要快速 HTML/JS 解析)
  - 新的 `crypto_analyzer` (加密分析模組)

**優勢**:
```rust
// 極高性能的模式匹配
use regex::Regex;
use rayon::prelude::*;

pub struct HighPerformanceScanner {
    patterns: Vec<Regex>,
}

impl HighPerformanceScanner {
    pub fn scan_parallel(&self, payloads: Vec<String>) -> Vec<Finding> {
        payloads.par_iter()
            .filter_map(|payload| self.detect_vulnerability(payload))
            .collect()
    }
}
```

### 🚀 **併發/網路密集 - Go**

**適用模組**:
- `function_authn_go` ✅ (已實現)
- `function_cspm_go` ✅ (已實現) 
- `function_ssrf_go` ✅ (已實現)
- **新增建議**:
  - `load_balancer` (負載均衡器)
  - `api_gateway` (API 閘道)
  - `health_monitor` (健康監控服務)

**優勢**:
```go
// 優秀的併發處理
func (s *Scanner) ProcessConcurrently(tasks []Task) []Result {
    const maxWorkers = 100
    taskCh := make(chan Task, len(tasks))
    resultCh := make(chan Result, len(tasks))
    
    // 啟動 worker goroutines
    for i := 0; i < maxWorkers; i++ {
        go s.worker(taskCh, resultCh)
    }
    
    // 分發任務
    for _, task := range tasks {
        taskCh <- task
    }
    close(taskCh)
    
    // 收集結果
    results := make([]Result, 0, len(tasks))
    for i := 0; i < len(tasks); i++ {
        results = append(results, <-resultCh)
    }
    
    return results
}
```

### 🌐 **前端互動/即時處理 - Node.js/TypeScript**

**適用模組**:
- `aiva_scan_node` ✅ (已實現)
- **新增建議**:
  - `realtime_dashboard` (即時儀表板)
  - `websocket_server` (WebSocket 服務)
  - `browser_automation` (瀏覽器自動化服務)

**優勢**:
```typescript
// 優秀的異步處理和瀏覽器整合
class RealtimeScanManager {
  private browser: Browser;
  private wsServer: WebSocketServer;
  
  async executeScanWithLiveUpdates(task: ScanTask): Promise<void> {
    const scan = await this.browser.newPage();
    
    // 即時回報掃描進度
    scan.on('response', (response) => {
      this.wsServer.broadcast({
        type: 'scan_progress',
        data: { url: response.url(), status: response.status() }
      });
    });
    
    // 非阻塞並行處理
    const results = await Promise.allSettled([
      this.extractContent(scan),
      this.simulateInteractions(scan),
      this.monitorNetworkActivity(scan)
    ]);
    
    return results;
  }
}
```

### 🧠 **業務邏輯/整合/AI - Python**

**保持模組**:
- `aiva_core` ✅ (核心業務邏輯)
- `aiva_integration` ✅ (系統整合)
- AI 相關功能 (策略生成、風險評估)
- 資料分析和報告生成

**優勢**:
```python
# 豐富的生態系統和快速開發
class AIEnhancedAnalyzer:
    def __init__(self):
        self.ml_model = joblib.load('vulnerability_classifier.pkl')
        self.nlp_processor = spacy.load('en_core_web_sm')
    
    async def intelligent_analysis(self, findings: List[Finding]) -> AnalysisResult:
        # 使用 ML 模型分類漏洞
        classifications = self.ml_model.predict([f.features for f in findings])
        
        # NLP 處理描述
        processed_descriptions = []
        for finding in findings:
            doc = self.nlp_processor(finding.description)
            processed_descriptions.append(self.extract_entities(doc))
        
        return AnalysisResult(
            classifications=classifications,
            entities=processed_descriptions,
            risk_score=self.calculate_risk_score(findings)
        )
```

## 📋 **具體遷移計劃**

### Phase 1: 高性能模組遷移至 Rust (8週)

```rust
// function_sqli_rust/src/lib.rs
use sqlparser::{ast::Statement, dialect::GenericDialect, parser::Parser};
use regex::RegexSet;

pub struct SqliDetector {
    dangerous_patterns: RegexSet,
    parser: Parser<GenericDialect>,
}

impl SqliDetector {
    pub fn new() -> Self {
        let patterns = vec![
            r"(?i)\bunion\s+select\b",
            r"(?i)\bor\s+\d+\s*=\s*\d+",
            r"(?i)'\s*or\s*'.*?'\s*=\s*'",
            // ... more patterns
        ];
        
        Self {
            dangerous_patterns: RegexSet::new(&patterns).unwrap(),
            parser: Parser::new(&GenericDialect {}),
        }
    }
    
    pub fn detect(&self, payload: &str) -> Option<SqliVulnerability> {
        // 高性能語法分析 + 模式匹配
        if self.dangerous_patterns.is_match(payload) {
            return Some(self.analyze_sql_structure(payload));
        }
        None
    }
}
```

### Phase 2: 微服務架構遷移至 Go (6週)

```go
// services/gateway/main.go
package main

import (
    "context"
    "net/http"
    "sync"
    "time"
    
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
)

type Gateway struct {
    services map[string]*ServicePool
    logger   *zap.Logger
    mu       sync.RWMutex
}

func (g *Gateway) RouteRequest(c *gin.Context) {
    service := g.selectHealthyService(c.Param("service"))
    if service == nil {
        c.JSON(503, gin.H{"error": "Service unavailable"})
        return
    }
    
    // 負載均衡 + 健康檢查
    ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
    defer cancel()
    
    result, err := service.Process(ctx, c.Request)
    if err != nil {
        g.handleServiceError(service, err)
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(200, result)
}
```

### Phase 3: 前端服務增強 (4週)

```typescript
// services/realtime-dashboard/src/scan-monitor.ts
import { WebSocketServer } from 'ws';
import { Browser } from 'playwright';

export class RealtimeScanMonitor {
  private wsServer: WebSocketServer;
  private activScans: Map<string, ScanSession> = new Map();
  
  constructor(port: number) {
    this.wsServer = new WebSocketServer({ port });
    this.setupWebSocketHandlers();
  }
  
  async startScan(scanId: string, config: ScanConfig): Promise<void> {
    const session = new ScanSession(scanId, config);
    this.activScans.set(scanId, session);
    
    // 即時進度廣播
    session.on('progress', (progress) => {
      this.broadcast({
        type: 'scan_progress',
        scanId,
        data: progress
      });
    });
    
    session.on('finding', (finding) => {
      this.broadcast({
        type: 'vulnerability_found',
        scanId,
        data: finding
      });
    });
    
    await session.execute();
  }
}
```

## 🔄 **服務間通信架構**

### 消息隊列設計
```yaml
# RabbitMQ 隊列規劃
queues:
  # 高頻掃描任務
  - name: "task.scan.fast"
    consumer: "rust_scanner"
    
  # 併發網路任務
  - name: "task.network.concurrent"
    consumer: "go_service"
    
  # 瀏覽器相關任務
  - name: "task.browser.dynamic"
    consumer: "node_service"
    
  # 複雜分析任務
  - name: "task.analysis.ai"
    consumer: "python_service"
```

### gRPC 服務定義
```protobuf
// services/proto/scanner.proto
syntax = "proto3";

service ScannerService {
  rpc ExecuteScan(ScanRequest) returns (stream ScanUpdate);
  rpc GetScanStatus(StatusRequest) returns (StatusResponse);
}

message ScanRequest {
  string scan_id = 1;
  string target_url = 2;
  ScanConfig config = 3;
}

message ScanUpdate {
  string scan_id = 1;
  UpdateType type = 2;
  bytes data = 3;
}
```

## 📊 **效益預期**

### 性能提升
- **Rust 模組**: 50-80% 性能提升
- **Go 微服務**: 3-5x 併發處理能力
- **Node.js 前端**: 實時響應 < 100ms

### 資源效率
- **記憶體使用**: -40% (Rust + Go 最佳化)
- **CPU 使用**: -30% (更好的併發模型)
- **部署大小**: -60% (編譯型語言)

### 開發效率
- **專業化分工**: 各團隊專精特定語言
- **模組化開發**: 獨立測試和部署
- **漸進式遷移**: 無需停機升級

## 🛣️ **實施時程**

| 階段 | 時程 | 重點工作 | 風險評估 |
|------|------|----------|----------|
| Phase 1 | 8週 | Rust 高性能模組 | 中等 - 學習曲線 |
| Phase 2 | 6週 | Go 微服務架構 | 低 - 成熟技術 |  
| Phase 3 | 4週 | Node.js 前端增強 | 低 - 現有基礎 |
| Phase 4 | 2週 | 整合測試優化 | 中等 - 系統複雜性 |

**總計**: 20週 (約 5個月)

## 💡 **建議**

1. **優先順序**: 先從性能瓶頸模組開始 (Rust)
2. **團隊培訓**: 提前安排 Rust/Go 技能培訓
3. **漸進遷移**: 保持舊版本並行運行
4. **監控指標**: 建立詳細的性能對比基準

這個多語言架構將讓 AIVA 在保持開發靈活性的同時，獲得各語言的最佳性能特性。