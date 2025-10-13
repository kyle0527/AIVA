# AIVA 完整實施指南

## 基於 AI 輔助開發的最佳實踐

**文件版本**: 1.0  
**建立日期**: 2025-10-13  
**目標**: 提供 AI 協助開發時的完整技術補充資訊  
**維護者**: AIVA 開發團隊

---

## 📋 目錄

1. [AI 輔助開發策略](#ai-輔助開發策略)
2. [完整技術棧清單](#完整技術棧清單)
3. [所需套件與依賴](#所需套件與依賴)
4. [開發環境設置](#開發環境設置)
5. [代碼生成範本](#代碼生成範本)
6. [測試策略與範例](#測試策略與範例)
7. [常見問題解決方案](#常見問題解決方案)
8. [AI Prompts 最佳實踐](#ai-prompts-最佳實踐)

---

## 🤖 AI 輔助開發策略

### 為什麼需要完整的技術補充?

使用 AI (如 GitHub Copilot, ChatGPT, Claude) 進行開發時,提供完整的上下文資訊可以:

1. **減少迭代次數**: AI 一次就能生成正確的代碼
2. **避免版本衝突**: 明確指定套件版本避免相容性問題
3. **統一代碼風格**: 提供範本確保團隊一致性
4. **加速問題解決**: 預先提供常見錯誤的解決方案

### AI 開發工作流程

```
需求定義 → 準備技術棧資訊 → 生成代碼 → 自動測試 → Code Review → 部署
   ↓            ↓                  ↓           ↓            ↓          ↓
[人工]     [本文件提供]        [AI 生成]   [自動化]     [AI 輔助]  [自動化]
```

---

## 📦 完整技術棧清單

### Python 環境

```toml
# pyproject.toml - 完整版本
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aiva"
version = "1.0.0"
description = "AIVA Intelligent Vulnerability Assessment System"
requires-python = ">=3.11"
dependencies = [
    # 核心框架
    "fastapi[all]==0.109.0",
    "uvicorn[standard]==0.27.0",
    "pydantic==2.5.3",
    "pydantic-settings==2.1.0",
    
    # 異步支援
    "asyncio==3.4.3",
    "aiohttp==3.9.1",
    "aiofiles==23.2.1",
    "httpx==0.26.0",
    
    # 數據庫
    "sqlalchemy[asyncio]==2.0.25",
    "asyncpg==0.29.0",
    "alembic==1.13.1",
    
    # 消息隊列
    "aio-pika==9.3.1",
    "celery[redis]==5.3.4",
    
    # 爬蟲相關
    "beautifulsoup4==4.12.2",
    "lxml==5.1.0",
    "selectolax==0.3.17",
    "playwright==1.41.0",
    
    # 安全工具
    "cryptography==42.0.0",
    "python-jose[cryptography]==3.3.0",
    "passlib[bcrypt]==1.7.4",
    
    # HTTP 客戶端
    "requests==2.31.0",
    "urllib3==2.1.0",
    
    # 日誌與監控
    "structlog==24.1.0",
    "python-json-logger==2.0.7",
    "opentelemetry-api==1.22.0",
    "opentelemetry-sdk==1.22.0",
    "opentelemetry-exporter-otlp==1.22.0",
    "prometheus-client==0.19.0",
    
    # 工具庫
    "python-multipart==0.0.6",
    "python-dotenv==1.0.0",
    "click==8.1.7",
    "rich==13.7.0",
    "pyyaml==6.0.1",
    "toml==0.10.2",
]

[project.optional-dependencies]
dev = [
    # 測試
    "pytest==7.4.4",
    "pytest-asyncio==0.23.3",
    "pytest-cov==4.1.0",
    "pytest-mock==3.12.0",
    "pytest-timeout==2.2.0",
    "faker==22.0.0",
    "factory-boy==3.3.0",
    
    # 代碼品質
    "ruff==0.1.14",
    "mypy==1.8.0",
    "bandit[toml]==1.7.6",
    "black==23.12.1",
    "isort==5.13.2",
    
    # 型別檢查
    "types-requests==2.31.0.20240106",
    "types-pyyaml==6.0.12.12",
    "types-toml==0.10.8.7",
    
    # 文檔
    "sphinx==7.2.6",
    "sphinx-rtd-theme==2.0.0",
    "myst-parser==2.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP", "B", "A", "C4", "DTZ", "T10", "DJ", "EM", "EXE", "ISC", "ICN", "G", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PD", "PGH", "PL", "TRY", "NPY", "RUF"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --cov=services --cov-report=html --cov-report=term"
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow running tests",
]
```

### Node.js 環境 (Playwright 掃描服務)

```json
// package.json
{
  "name": "aiva-scan-node",
  "version": "1.0.0",
  "description": "AIVA Dynamic Scanning Engine with Playwright",
  "main": "dist/index.js",
  "type": "module",
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=10.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "lint": "eslint src --ext .ts",
    "format": "prettier --write \"src/**/*.ts\""
  },
  "dependencies": {
    "@grpc/grpc-js": "^1.10.0",
    "@grpc/proto-loader": "^0.7.13",
    "playwright": "^1.41.0",
    "@opentelemetry/api": "^1.8.0",
    "@opentelemetry/sdk-node": "^0.48.0",
    "@opentelemetry/exporter-trace-otlp-grpc": "^0.48.0",
    "pino": "^8.17.0",
    "pino-pretty": "^10.3.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "eslint": "^8.56.0",
    "prettier": "^3.2.0",
    "tsx": "^4.7.0",
    "typescript": "^5.3.3",
    "vitest": "^1.2.0"
  }
}
```

### Go 環境 (SSRF/SQLi 探測器)

```go
// go.mod
module github.com/aiva/function-go

go 1.21

require (
    google.golang.org/grpc v1.61.0
    google.golang.org/protobuf v1.32.0
    
    // OpenTelemetry
    go.opentelemetry.io/otel v1.22.0
    go.opentelemetry.io/otel/sdk v1.22.0
    go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.22.0
    
    // 日誌
    go.uber.org/zap v1.26.0
    
    // HTTP 客戶端
    github.com/go-resty/resty/v2 v2.11.0
    
    // 測試
    github.com/stretchr/testify v1.8.4
    github.com/golang/mock v1.6.0
)
```

### Rust 環境 (敏感資訊掃描器)

```toml
# Cargo.toml
[package]
name = "aiva-info-gatherer"
version = "1.0.0"
edition = "2021"

[dependencies]
# gRPC
tonic = "0.11"
prost = "0.12"
tokio = { version = "1.35", features = ["full"] }
tokio-stream = "0.1"

# 正則引擎
regex = "1.10"
aho-corasick = "1.1"

# 並行處理
rayon = "1.8"

# 日誌
tracing = "0.1"
tracing-subscriber = "0.3"

# OpenTelemetry
opentelemetry = "0.21"
opentelemetry-otlp = "0.14"

[dev-dependencies]
criterion = "0.5"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

---

## 🛠️ 開發環境設置

### 1. Python 開發環境

```bash
# Windows PowerShell
# 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝依賴
pip install --upgrade pip setuptools wheel
pip install -e .[dev]

# 設置 pre-commit hooks
pip install pre-commit
pre-commit install

# 驗證安裝
pytest --version
ruff --version
mypy --version
```

```bash
# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e .[dev]
pre-commit install
```

### 2. Node.js 開發環境

```bash
# 安裝 Node.js 20+ (使用 nvm)
nvm install 20
nvm use 20

# 初始化專案
cd services/scan/aiva_scan_node
npm install

# 安裝 Playwright 瀏覽器
npx playwright install --with-deps chromium firefox webkit

# 驗證
npm run build
npm test
```

### 3. Go 開發環境

```bash
# 安裝 Go 1.21+
# Windows: https://go.dev/dl/
# Linux: sudo snap install go --classic

# 初始化專案
cd services/function/function_ssrf_go
go mod download

# 安裝工具
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 驗證
go test ./...
```

### 4. Rust 開發環境

```bash
# 安裝 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 初始化專案
cd services/scan/info_gatherer_rust
cargo build --release

# 驗證
cargo test
cargo bench
```

### 5. Docker 開發環境

```yaml
# docker-compose.dev.yml - 完整開發環境
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: aiva
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: aiva_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: aiva
      RABBITMQ_DEFAULT_PASS: dev_password
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
  
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.93.0
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./docker/otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
  
  jaeger:
    image: jaegertracing/all-in-one:1.53
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
  
  prometheus:
    image: prom/prometheus:v2.48.0
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:10.2.3
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"

volumes:
  postgres_data:
  rabbitmq_data:
  prometheus_data:
  grafana_data:
```

---

## 📝 代碼生成範本

### Python 範本

#### 1. 測試範本

```python
# tests/conftest.py - 全域 fixtures
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from services.aiva_common.schemas import *

# 測試數據庫
TEST_DATABASE_URL = "postgresql+asyncpg://aiva:test@localhost/aiva_test"

@pytest.fixture(scope="session")
def event_loop():
    """創建 event loop"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def engine():
    """創建數據庫引擎"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()

@pytest.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """創建數據庫 session"""
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def sample_asset() -> Asset:
    """範例 Asset"""
    return Asset(
        type="url",
        value="https://example.com/api/users/123",
        metadata={
            "method": "GET",
            "status_code": 200,
            "content_type": "application/json"
        }
    )

@pytest.fixture
def sample_vulnerability() -> Vulnerability:
    """範例 Vulnerability"""
    return Vulnerability(
        type=VulnerabilityType.XSS,
        severity=Severity.HIGH,
        confidence=0.85,
        description="Reflected XSS in search parameter",
        evidence={"payload": "<script>alert(1)</script>"}
    )
```

```python
# tests/core/test_strategy_generator.py
import pytest
from services.core.aiva_core.analysis.strategy_generator import StrategyGenerator
from services.aiva_common.schemas import Asset, Vulnerability
from services.aiva_common.enums import VulnerabilityType, Severity

@pytest.fixture
def strategy_generator():
    return StrategyGenerator()

@pytest.fixture
def sample_assets():
    return [
        Asset(type="url", value="https://example.com/admin", metadata={"sensitive": True}),
        Asset(type="url", value="https://example.com/api/users?id=123", metadata={"has_param": True}),
        Asset(type="form", value="login_form", metadata={"method": "POST", "action": "/login"}),
    ]

class TestStrategyGenerator:
    @pytest.mark.asyncio
    async def test_generate_strategy_with_no_vulnerabilities(
        self, strategy_generator, sample_assets
    ):
        """測試無已知漏洞時的策略生成"""
        strategy = await strategy_generator.generate_strategy(sample_assets, [])
        
        assert strategy is not None
        assert len(strategy.tests) > 0
        assert all(test.priority >= 1 for test in strategy.tests)
    
    @pytest.mark.asyncio
    async def test_generate_strategy_with_known_sqli(
        self, strategy_generator, sample_assets
    ):
        """測試已知 SQLi 漏洞時的策略調整"""
        vulnerabilities = [
            Vulnerability(
                type=VulnerabilityType.SQLI,
                severity=Severity.HIGH,
                confidence=0.9,
                description="SQL Injection in user parameter"
            )
        ]
        
        strategy = await strategy_generator.generate_strategy(sample_assets, vulnerabilities)
        
        # 應該優先測試 SQLi
        sqli_tests = [t for t in strategy.tests if t.module == "sqli"]
        assert len(sqli_tests) > 0
        assert sqli_tests[0].priority >= 8
    
    @pytest.mark.asyncio
    async def test_priority_calculation(self, strategy_generator, sample_assets):
        """測試優先級計算邏輯"""
        strategy = await strategy_generator.generate_strategy(sample_assets, [])
        
        # 敏感路徑應該有更高優先級
        admin_tests = [t for t in strategy.tests if "admin" in t.target.url]
        regular_tests = [t for t in strategy.tests if "admin" not in t.target.url]
        
        if admin_tests and regular_tests:
            assert max(t.priority for t in admin_tests) >= max(t.priority for t in regular_tests)
```

#### 2. Service 層範本

```python
# services/core/aiva_core/services/scan_service.py
from typing import List, Optional
from services.aiva_common.schemas import ScanStartPayload, ScanCompletedPayload, Asset
from services.aiva_common.enums import ModuleName, MessageType
from services.core.aiva_core.analysis.attack_surface_analyzer import AttackSurfaceAnalyzer
from services.core.aiva_core.analysis.strategy_generator import StrategyGenerator
from services.core.aiva_core.execution.task_generator import TaskGenerator
import structlog

logger = structlog.get_logger(__name__)

class ScanService:
    """掃描服務 - 協調整個掃描流程"""
    
    def __init__(self):
        self.surface_analyzer = AttackSurfaceAnalyzer()
        self.strategy_generator = StrategyGenerator()
        self.task_generator = TaskGenerator()
    
    async def process_scan_completed(
        self, 
        scan_id: str, 
        payload: ScanCompletedPayload
    ) -> List[tuple[str, dict]]:
        """
        處理掃描完成事件
        
        Args:
            scan_id: 掃描 ID
            payload: 掃描完成載荷
        
        Returns:
            要發布的任務列表 [(topic, payload), ...]
        """
        logger.info(
            "Processing scan completed",
            scan_id=scan_id,
            asset_count=len(payload.assets),
            vulnerability_count=len(payload.vulnerabilities)
        )
        
        try:
            # 1. 分析攻擊面
            attack_surface = await self.surface_analyzer.analyze(
                payload.assets,
                payload.vulnerabilities
            )
            logger.info(
                "Attack surface analyzed",
                scan_id=scan_id,
                xss_candidates=len(attack_surface.xss_candidates),
                sqli_candidates=len(attack_surface.sqli_candidates),
                ssrf_candidates=len(attack_surface.ssrf_candidates),
                idor_candidates=len(attack_surface.idor_candidates)
            )
            
            # 2. 生成測試策略
            strategy = await self.strategy_generator.generate_strategy(
                payload.assets,
                payload.vulnerabilities
            )
            logger.info(
                "Test strategy generated",
                scan_id=scan_id,
                total_tests=len(strategy.tests),
                estimated_duration=strategy.estimated_duration_seconds
            )
            
            # 3. 生成具體任務
            tasks = await self.task_generator.generate_from_strategy(
                scan_id,
                strategy,
                attack_surface
            )
            logger.info(
                "Tasks generated",
                scan_id=scan_id,
                task_count=len(tasks)
            )
            
            return tasks
            
        except Exception as e:
            logger.error(
                "Failed to process scan completed",
                scan_id=scan_id,
                error=str(e),
                exc_info=True
            )
            raise
```

### Node.js 範本

```typescript
// src/services/scan-service.ts
import { chromium, Browser, Page, BrowserContext } from 'playwright';
import { ScanTask, Finding, Severity } from '../generated/aiva/v1/scan_pb';
import { logger } from '../utils/logger';
import { tracer } from '../utils/telemetry';

export class ScanService {
  private browser: Browser | null = null;
  
  async initialize(): Promise<void> {
    this.browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    logger.info('Browser initialized');
  }
  
  async scanXSS(task: ScanTask): Promise<Finding[]> {
    const span = tracer.startSpan('scanXSS');
    span.setAttribute('task.id', task.taskId);
    span.setAttribute('task.target', task.target);
    
    const findings: Finding[] = [];
    let context: BrowserContext | null = null;
    let page: Page | null = null;
    
    try {
      context = await this.browser!.newContext({
        viewport: { width: 1920, height: 1080 },
        userAgent: 'AIVA-Scanner/1.0'
      });
      
      page = await context.newPage();
      
      // 設置 XSS 檢測
      await page.addInitScript(() => {
        window.__AIVA_XSS_DETECTED__ = false;
        
        const originalSetAttribute = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function(name: string, value: string) {
          if (name === 'src' && value.includes('<script>')) {
            window.__AIVA_XSS_DETECTED__ = true;
          }
          return originalSetAttribute.call(this, name, value);
        };
      });
      
      await page.goto(task.target);
      
      // 測試 XSS payloads
      const payloads = [
        '<script>alert(1)</script>',
        '<img src=x onerror=alert(1)>',
        '"><script>alert(1)</script>'
      ];
      
      const inputs = await page.locator('input[type="text"], textarea').all();
      
      for (const input of inputs) {
        for (const payload of payloads) {
          await input.fill(payload);
          await page.keyboard.press('Enter');
          await page.waitForTimeout(1000);
          
          const xssDetected = await page.evaluate(() => window.__AIVA_XSS_DETECTED__);
          
          if (xssDetected) {
            const finding = new Finding({
              taskId: task.taskId,
              module: 'xss',
              severity: Severity.SEV_HIGH,
              title: 'Reflected XSS Detected',
              summary: `Payload: ${payload}`,
              evidence: Buffer.from(await page.content())
            });
            findings.push(finding);
            logger.warn('XSS detected', { taskId: task.taskId, payload });
          }
        }
      }
      
      span.setStatus({ code: 0 });
      
    } catch (error) {
      span.setStatus({ code: 2, message: error.message });
      logger.error('XSS scan failed', { error: error.message });
      throw error;
      
    } finally {
      if (page) await page.close();
      if (context) await context.close();
      span.end();
    }
    
    return findings;
  }
  
  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
  }
}
```

### Go 範本

```go
// internal/detector/ssrf_detector.go
package detector

import (
    "context"
    "fmt"
    "net"
    "net/http"
    "time"
    
    pb "github.com/aiva/function-go/proto/v1"
    "go.uber.org/zap"
)

type SSRFDetector struct {
    logger        *zap.Logger
    client        *http.Client
    blockedRanges []*net.IPNet
}

func NewSSRFDetector(logger *zap.Logger) *SSRFDetector {
    // 阻擋的 IP 範圍
    blockedCIDRs := []string{
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.169.254/32",  // AWS IMDS
    }
    
    var ranges []*net.IPNet
    for _, cidr := range blockedCIDRs {
        _, ipNet, _ := net.ParseCIDR(cidr)
        ranges = append(ranges, ipNet)
    }
    
    client := &http.Client{
        Timeout: 5 * time.Second,
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            if len(via) >= 3 {
                return fmt.Errorf("too many redirects")
            }
            return nil
        },
    }
    
    return &SSRFDetector{
        logger:        logger,
        client:        client,
        blockedRanges: ranges,
    }
}

func (d *SSRFDetector) Scan(ctx context.Context, task *pb.ScanTask) ([]*pb.Finding, error) {
    d.logger.Info("Starting SSRF scan", zap.String("task_id", task.TaskId))
    
    findings := []*pb.Finding{}
    
    // SSRF payloads
    payloads := []string{
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1:8080/admin",
    }
    
    for _, payload := range payloads {
        targetURL := fmt.Sprintf("%s?url=%s", task.Target, payload)
        
        req, err := http.NewRequestWithContext(ctx, "GET", targetURL, nil)
        if err != nil {
            d.logger.Error("Failed to create request", zap.Error(err))
            continue
        }
        
        resp, err := d.client.Do(req)
        if err != nil {
            d.logger.Debug("Request failed", zap.String("payload", payload), zap.Error(err))
            continue
        }
        resp.Body.Close()
        
        if resp.StatusCode == 200 {
            finding := &pb.Finding{
                TaskId:   task.TaskId,
                Module:   "ssrf",
                Severity: pb.Severity_SEV_HIGH,
                Title:    "SSRF Vulnerability Detected",
                Summary:  fmt.Sprintf("Successful request to: %s", payload),
                Evidence: []byte(fmt.Sprintf("Status: %d", resp.StatusCode)),
            }
            findings = append(findings, finding)
            d.logger.Warn("SSRF detected", zap.String("payload", payload))
        }
    }
    
    return findings, nil
}

func (d *SSRFDetector) isPrivateIP(ip net.IP) bool {
    for _, blocked := range d.blockedRanges {
        if blocked.Contains(ip) {
            return true
        }
    }
    return false
}
```

---

## 🧪 測試策略與範例

### 測試金字塔

```
        /\
       /E2E\      <- 10% (端到端測試)
      /------\
     /Integr.\   <- 20% (整合測試)
    /----------\
   / Unit Tests \  <- 70% (單元測試)
  /--------------\
```

### Python 測試範例

```python
# tests/core/test_attack_surface_analyzer.py
import pytest
from services.core.aiva_core.analysis.attack_surface_analyzer import AttackSurfaceAnalyzer
from services.aiva_common.schemas import Asset, Vulnerability
from services.aiva_common.enums import VulnerabilityType

class TestAttackSurfaceAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return AttackSurfaceAnalyzer()
    
    @pytest.fixture
    def url_assets(self):
        return [
            Asset(type="url", value="https://example.com/search?q=test", metadata={}),
            Asset(type="url", value="https://example.com/api/users/123", metadata={}),
            Asset(type="form", value="login_form", metadata={"action": "/login", "method": "POST"}),
        ]
    
    @pytest.mark.asyncio
    async def test_detect_xss_candidates(self, analyzer, url_assets):
        """測試 XSS 候選檢測"""
        surface = await analyzer.analyze(url_assets, [])
        
        # 應該檢測到包含查詢參數的 URL
        assert len(surface.xss_candidates) > 0
        assert any("search" in c.asset.value for c in surface.xss_candidates)
    
    @pytest.mark.asyncio
    async def test_detect_sqli_candidates(self, analyzer, url_assets):
        """測試 SQLi 候選檢測"""
        surface = await analyzer.analyze(url_assets, [])
        
        # 應該檢測到包含 ID 參數的 URL
        assert len(surface.sqli_candidates) > 0
        assert any("users/123" in c.asset.value for c in surface.sqli_candidates)
    
    @pytest.mark.asyncio
    async def test_high_risk_asset_detection(self, analyzer):
        """測試高風險資產檢測"""
        admin_assets = [
            Asset(type="url", value="https://example.com/admin/users", metadata={}),
            Asset(type="url", value="https://example.com/config.json", metadata={}),
        ]
        
        surface = await analyzer.analyze(admin_assets, [])
        
        # admin 和 config 應該被標記為高風險
        assert len(surface.high_risk_assets) == 2
```

### Node.js 測試範例

```typescript
// tests/scan-service.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { ScanService } from '../src/services/scan-service';
import { ScanTask } from '../src/generated/aiva/v1/scan_pb';

describe('ScanService', () => {
  let scanService: ScanService;
  
  beforeAll(async () => {
    scanService = new ScanService();
    await scanService.initialize();
  });
  
  afterAll(async () => {
    await scanService.cleanup();
  });
  
  it('should detect reflected XSS', async () => {
    const task = new ScanTask({
      taskId: 'test-123',
      module: 'xss',
      target: 'http://testphp.vulnweb.com/search.php?test=query',
    });
    
    const findings = await scanService.scanXSS(task);
    
    expect(findings).toBeDefined();
    expect(findings.length).toBeGreaterThan(0);
  });
  
  it('should handle scan timeout gracefully', async () => {
    const task = new ScanTask({
      taskId: 'test-456',
      module: 'xss',
      target: 'http://httpstat.us/200?sleep=10000',
    });
    
    await expect(scanService.scanXSS(task)).rejects.toThrow();
  });
});
```

### Go 測試範例

```go
// internal/detector/ssrf_detector_test.go
package detector

import (
    "context"
    "testing"
    "time"
    
    pb "github.com/aiva/function-go/proto/v1"
    "github.com/stretchr/testify/assert"
    "go.uber.org/zap"
)

func TestSSRFDetector_Scan(t *testing.T) {
    logger, _ := zap.NewDevelopment()
    detector := NewSSRFDetector(logger)
    
    t.Run("should detect SSRF to IMDS", func(t *testing.T) {
        task := &pb.ScanTask{
            TaskId: "test-123",
            Module: "ssrf",
            Target: "http://vulnerable-site.com/fetch",
        }
        
        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
        defer cancel()
        
        findings, err := detector.Scan(ctx, task)
        
        assert.NoError(t, err)
        assert.NotEmpty(t, findings)
    })
    
    t.Run("should block private IP ranges", func(t *testing.T) {
        privateIP := "192.168.1.1"
        ip := net.ParseIP(privateIP)
        
        isPrivate := detector.isPrivateIP(ip)
        
        assert.True(t, isPrivate)
    })
}
```

---

## 🔧 常見問題解決方案

### 問題 1: Playwright 在 Docker 中無法啟動

**錯誤訊息**:

```
Error: browserType.launch: Failed to launch chromium because executable doesn't exist
```

**解決方案**:

```dockerfile
# Dockerfile
FROM mcr.microsoft.com/playwright:v1.41.0-jammy

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

# 確保瀏覽器已安裝
RUN npx playwright install --with-deps chromium

CMD ["node", "dist/index.js"]
```

### 問題 2: gRPC Python 生成代碼找不到

**錯誤訊息**:

```
ModuleNotFoundError: No module named 'aiva.v1.scan_pb2'
```

**解決方案**:

```bash
# 確保 proto 編譯正確
python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=services/aiva_common/generated/python \
  --grpc_python_out=services/aiva_common/generated/python \
  proto/aiva/v1/*.proto

# 創建 __init__.py
touch services/aiva_common/generated/python/aiva/__init__.py
touch services/aiva_common/generated/python/aiva/v1/__init__.py
```

### 問題 3: OpenTelemetry 追蹤無法串接

**問題**: Python → Node.js 的 trace 斷掉

**解決方案**:

```python
# Python 端 - 傳遞 trace context
from opentelemetry import trace, context
from opentelemetry.propagate import inject

# 注入 context 到 metadata
metadata = []
inject(metadata)

async with grpc.aio.insecure_channel('localhost:50051') as channel:
    stub = ScanServiceStub(channel)
    await stub.StreamFindings(task, metadata=metadata)
```

```typescript
// Node.js 端 - 提取 trace context
import { context, propagation } from '@opentelemetry/api';

const extractedContext = propagation.extract(context.active(), metadata);
const span = tracer.startSpan('scanXSS', {}, extractedContext);
```

---

## 💡 AI Prompts 最佳實踐

### 高效 Prompt 範本

#### 1. 生成測試代碼

```
請為以下 Python 類別生成完整的單元測試:

**目標類別**:
[貼上代碼]

**要求**:
- 使用 pytest 框架
- 涵蓋正常流程、邊界條件、異常處理
- 使用 pytest.fixture 管理測試數據
- 使用 pytest.mark.asyncio 處理異步函數
- 測試覆蓋率 >90%
- 包含清晰的測試命名和文檔字串

**依賴套件**:
- pytest==7.4.4
- pytest-asyncio==0.23.3
- pytest-mock==3.12.0

**現有 Schema 定義**:
[貼上相關 Schema]
```

#### 2. 重構現有代碼

```
請重構以下代碼,改進可維護性和性能:

**原始代碼**:
[貼上代碼]

**目標**:
1. 提取重複邏輯為獨立函數
2. 改善錯誤處理 (使用 try-except-else-finally)
3. 添加型別提示 (Python 3.11+)
4. 優化性能瓶頸
5. 添加詳細的 docstring

**技術棧**:
- Python 3.11
- Pydantic v2
- asyncio
- structlog

**保持**:
- 現有的公開 API 介面不變
- 向後兼容性
```

#### 3. 實作新功能

```
請實作以下功能:

**功能描述**:
基於規則引擎的策略生成器,根據攻擊面和已知漏洞生成測試任務

**技術要求**:
- Python 3.11 async/await
- 使用 Pydantic BaseModel 定義數據結構
- 規則引擎基於簡單的 if-elif-else
- 優先級計算考慮:
  1. 資產敏感度 (admin, config 等)
  2. 已知漏洞類型
  3. 參數複雜度
  4. 歷史成功率

**輸入**:
- List[Asset]: 已發現的資產
- List[Vulnerability]: 已知漏洞

**輸出**:
- TestStrategy: 包含排序後的測試任務列表

**參考 Schema**:
[貼上 Asset, Vulnerability, TestStrategy 定義]

**測試需求**:
同時生成對應的 pytest 測試代碼
```

### AI 輔助 Debug Prompt

```
我遇到以下錯誤:

**錯誤訊息**:
[貼上完整錯誤堆疊]

**相關代碼**:
[貼上出錯的代碼片段]

**環境資訊**:
- Python 3.11
- 作業系統: Windows 11
- 相關套件版本:
  - pydantic==2.5.3
  - fastapi==0.109.0

**已嘗試的解決方案**:
1. [描述]
2. [描述]

請提供:
1. 錯誤原因分析
2. 具體的修復代碼
3. 如何避免類似問題的最佳實踐
```

---

## 📚 附錄

### A. VS Code 推薦配置

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.extraPaths": [
    "${workspaceFolder}/services"
  ],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.rulers": [100]
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[go]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "golang.go"
  }
}
```

### B. Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]
```

---

**文件結束**  
**下次更新**: 依實際開發進度調整  
**維護者**: AIVA 開發團隊
