# AIVA Scan - 多語言統一掃描引擎

AIVA Scan 是一個強大的多語言統一掃描引擎，整合了 Python、TypeScript 和 Rust 三種技術的優勢，提供全方位的網路安全掃描能力。

---

## 🔧 修復原則

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些函數可能是：
- 預留的 API 端點或介面
- 未來功能的基礎架構
- 測試或除錯用途的輔助函數
- 向下相容性考量的舊版介面

說不定未來會用到，保持程式碼的擴展性和靈活性。

---

## � 目錄

- [🛠️ Scan 模組開發工具](#️-scan-模組開發工具)
- [🏗️ 整體架構](#️-整體架構)
- [📁 目錄結構](#-目錄結構)
- [🚀 現有功能](#-現有功能)
  - [1. Python 核心掃描引擎](#1-python-核心掃描引擎-aiva_scan)
  - [2. TypeScript 動態引擎](#2-typescript-動態引擎-aiva_scan_node)
  - [3. Rust 資訊收集器](#3-rust-資訊收集器-info_gatherer_rust)
  - [4. 統一掃描引擎](#4-統一掃描引擎-unified_scan_enginepy)
  - [5. SARIF 轉換器](#5-sarif-轉換器-sarif_converterpy)
- [💻 如何使用](#-如何使用)
- [🔌 技術整合](#-技術整合)
- [🎯 未來規劃](#-未來規劃)
- [📊 效能指標](#-效能指標)
- [🔐 安全性考量](#-安全性考量)
- [🛠️ 開發指南](#️-開發指南)

---

## �🛠️ Scan 模組開發工具

> **Python + TypeScript + Rust 混合架構**: 根據不同子模組選擇對應工具

| 子模組 | 主要語言 | 推薦插件 | 工具連結 |
|-------|---------|---------|---------|
| 🐍 **aiva_scan/** | Python | Pylance + Ruff + Black | [Python 工具 (22個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#1-python-開發生態-22-個) |
| 📘 **aiva_scan_node/** | TypeScript | ESLint + Prettier | [TypeScript 工具](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#7-程式碼品質與-linting-5-個) |
| 🦀 **info_gatherer_rust/** | Rust | rust-analyzer | [Rust 工具](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#3-其他程式語言-5-個) |

### 掃描模組專用工具

| 功能需求 | 推薦插件 | 說明 |
|---------|---------|------|
| 🌐 **網路除錯** | REST Client | 測試 HTTP 請求、API 掃描驗證 |
| 🔍 **安全分析** | SonarLint (4.33.0) | 靜態安全分析、漏洞檢測 |
| 🤖 **AI 輔助** | GitHub Copilot | 掃描邏輯生成、正則表達式協助 |
| 📊 **結果分析** | Rainbow CSV + Jupyter | 掃描結果資料分析 |
| 🐳 **環境管理** | Docker + Remote Containers | 隔離測試環境 |
| 🔀 **版本控制** | GitLens + Git Graph | 追蹤掃描邏輯變更 |

📚 **完整工具清單**: [VS Code 插件參考](../../_out/VSCODE_EXTENSIONS_INVENTORY.md) (88個插件)

### 💡 Scan 開發快速技巧

**Python 掃描引擎 (aiva_scan/)**:
- 使用 Pylance 進行型別檢查，確保掃描器參數正確
- Ruff 快速檢測常見錯誤（如未處理的異常）
- autoDocstring 生成掃描器說明文檔

**TypeScript 動態引擎 (aiva_scan_node/)**:
- ESLint 檢查異步程式碼問題
- Prettier 統一格式化
- 使用 Code Runner 快速測試 Playwright 腳本

**Rust 資訊收集器 (info_gatherer_rust/)**:
- rust-analyzer 提供即時編譯檢查
- Cargo 任務整合測試與建構
- ErrorLens 即時顯示編譯錯誤

**掃描測試**:
```bash
# 使用 REST Client 測試掃描 API
# 建立 scan.http 檔案
POST http://localhost:8000/scan/start
Content-Type: application/json

{
  "target": "https://example.com",
  "scan_type": "vulnerability"
}
```

**除錯技巧**: [多語言除錯指南](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-問題排查流程) | [掃描器最佳實踐](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-實際使用案例)

---

## 🏗️ 整體架構

```mermaid
graph TB
    subgraph "AIVA Scan 統一掃描引擎"
        direction TB
        
        subgraph "協調層 | Orchestration Layer"
            USG[統一掃描引擎<br/>UnifiedScanEngine]
            SO[掃描編排器<br/>ScanOrchestrator]
        end

        subgraph "Python 掃描引擎 | aiva_scan/"
            direction LR
            SC[掃描上下文<br/>ScanContext]
            CCE[核心爬取引擎<br/>CoreCrawlingEngine]
            VS[漏洞掃描器<br/>VulnerabilityScanner]
            AM[認證管理<br/>AuthManager]
            FM[指紋管理<br/>FingerprintManager]
            NSC[網路掃描器<br/>NetworkScanner]
            SDT[服務檢測<br/>ServiceDetector]
        end

        subgraph "TypeScript 動態引擎 | aiva_scan_node/"
            direction LR
            PIS[Phase-I 整合服務<br/>PhaseIService]
            PW[Playwright 引擎<br/>Browser Automation]
            CSA[客戶端授權檢測<br/>Client Auth Bypass]
            JS[JavaScript 分析器<br/>JS Analyzer]
        end

        subgraph "Rust 資訊收集器 | info_gatherer_rust/"
            direction LR
            SEC[敏感資訊檢測<br/>Secret Scanner]
            GH[Git 歷史掃描<br/>Git History]
            API[API 密鑰檢測<br/>API Key Scanner]
            REG[正則引擎<br/>Regex Engine]
        end

        subgraph "共享模組 | Shared Components"
            SCHEMA[統一 Schema<br/>Unified Models]
            CONFIG[配置管理<br/>Configuration]
            MQ[消息隊列<br/>Message Queue]
            LOGS[日誌系統<br/>Logging]
        end
    end

    %% 連接關係
    USG --> SO
    SO --> SC
    SC --> CCE
    SC --> VS
    CCE --> AM
    CCE --> FM
    
    USG --> PIS
    PIS --> PW
    PIS --> CSA
    PW --> JS
    
    USG --> SEC
    SEC --> GH
    SEC --> API
    SEC --> REG
    
    SO -.-> MQ
    PIS -.-> MQ
    SEC -.-> LOGS
    
    %% 樣式
    classDef pythonStyle fill:#3776ab,stroke:#2d5aa0,stroke-width:2px,color:#fff
    classDef typescriptStyle fill:#3178c6,stroke:#2b6cb0,stroke-width:2px,color:#fff
    classDef rustStyle fill:#ce422b,stroke:#a0351f,stroke-width:2px,color:#fff
    classDef sharedStyle fill:#6b7280,stroke:#4b5563,stroke-width:2px,color:#fff
    
    class SO,SC,CCE,VS,AM,FM,NSC,SDT pythonStyle
    class PIS,PW,CSA,JS typescriptStyle
    class SEC,GH,API,REG rustStyle
    class USG,SCHEMA,CONFIG,MQ,LOGS sharedStyle
```

## 📁 目錄結構

```
services/scan/
├── __init__.py                    # 模組入口點
├── README.md                     # 本文檔
├── unified_scan_engine.py        # 🎯 統一掃描引擎
├── schemas.py                    # 掃描相關 Schema
├── models.py                     # 資料模型定義
├── discovery_schemas.py          # 發現模組 Schema
│
├── aiva_scan/                    # 🐍 Python 核心掃描引擎
│   ├── __init__.py
│   ├── scan_orchestrator.py     # 掃描編排器 (核心)
│   ├── scan_context.py          # 掃描上下文管理
│   ├── strategy_controller.py   # 策略控制器
│   ├── worker.py                # 工作程序
│   │
│   ├── core_crawling_engine/    # 核心爬取引擎
│   │   ├── http_client_hi.py
│   │   ├── static_content_parser.py
│   │   └── url_queue_manager.py
│   │
│   ├── dynamic_engine/          # 動態掃描引擎
│   │   ├── dynamic_content_extractor.py
│   │   └── headless_browser_pool.py
│   │
│   ├── info_gatherer/          # 資訊收集器
│   │   ├── javascript_source_analyzer.py
│   │   └── sensitive_info_detector.py
│   │
│   ├── authentication_manager.py # 認證管理
│   ├── fingerprint_manager.py   # 指紋識別
│   ├── vulnerability_scanner.py # 漏洞掃描
│   ├── network_scanner.py       # 網路掃描
│   ├── service_detector.py      # 服務檢測
│   └── examples/               # 使用範例
│
├── aiva_scan_node/             # 🟨 TypeScript 動態掃描引擎
│   ├── package.json
│   ├── tsconfig.json
│   ├── README.md
│   ├── phase-i-integration.service.ts # Phase-I 整合服務
│   ├── src/                    # 源代碼
│   │   ├── services/
│   │   ├── interfaces/
│   │   └── utils/
│   └── types/                  # 型別定義
│       └── playwright.d.ts
│
└── info_gatherer_rust/        # 🦀 Rust 高性能資訊收集器
    ├── Cargo.toml
    ├── Cargo.lock
    ├── README.md
    ├── src/
    │   ├── main.rs
    │   ├── scanner.rs
    │   └── patterns.rs
    └── target/               # 編譯輸出
```

## 🚀 現有功能

### 1. Python 核心掃描引擎 (aiva_scan/)

#### 核心功能
- **ScanOrchestrator**: 統一掃描編排，支援多種掃描策略
- **策略控制**: 支援 FAST/COMPREHENSIVE/AGGRESSIVE 三種掃描模式
- **動態掃描**: 整合 Headless Browser 進行 JavaScript 渲染
- **認證管理**: 支援多種認證方式 (Basic, Bearer, Custom Headers)
- **指紋識別**: 自動識別技術棧和服務版本

#### 掃描能力
- **網路掃描**: 端口掃描、服務檢測、橫幅抓取
- **漏洞掃描**: SQL 注入、XSS、CSRF、目錄遍歷等
- **內容分析**: 靜態內容解析、JavaScript 源碼分析
- **敏感資訊**: API 密鑰、密碼、憑證等檢測

### 2. TypeScript 動態引擎 (aiva_scan_node/)

#### Phase-I 高價值功能
- **客戶端授權繞過檢測**: 檢測前端權限控制漏洞
- **Playwright 自動化**: 真實瀏覽器環境掃描
- **JavaScript 深度分析**: 代碼邏輯分析和漏洞挖掘
- **DOM 操作檢測**: 動態內容變化監控

#### 技術特性
- **TypeScript 型別安全**: 完整的型別定義
- **非同步處理**: 高效的並發掃描
- **模組化架構**: 易於擴展和維護

### 3. Rust 資訊收集器 (info_gatherer_rust/)

#### 性能優勢
- **極致性能**: 比 Python 快 10-100 倍
- **低記憶體消耗**: 單次掃描僅需 ~5 MB
- **並行處理**: Rayon 並行引擎
- **零拷貝**: 高效字串處理

#### 檢測類型
- AWS Access/Secret Keys
- GitHub Tokens  
- API Keys
- Private Keys (PEM/SSH)
- JWT Tokens
- Database Connection Strings
- Email 地址和 IP 位址

### 4. 統一掃描引擎 (unified_scan_engine.py)

#### 整合能力
- **多引擎協調**: 統一管理三種技術引擎
- **Phase-I 整合**: 高價值功能模組整合
- **策略配置**: 彈性的掃描策略配置
- **結果聚合**: 統一的結果格式和匯報
- **SARIF 2.1.0 輸出**: 支援業界標準的安全報告格式

### 5. SARIF 轉換器 (sarif_converter.py)

#### SARIF 2.1.0 標準支援
AIVA Scan 完全支援 SARIF (Static Analysis Results Interchange Format) 2.1.0 標準，可與以下工具整合：
- **GitHub Security**: 自動顯示在 GitHub Security Code Scanning
- **Azure DevOps**: 整合至 Azure Pipelines 安全掃描
- **VS Code**: 直接在編輯器中顯示漏洞
- **其他 SARIF 相容工具**: SonarQube、Checkmarx 等

#### 功能特性
- **自動映射嚴重程度**: 將 CVSS 評分映射到 SARIF level (error/warning/note)
- **完整證據鏈**: 包含請求、響應、載荷等完整證據
- **標準化規則**: 支援 CWE/CVE 引用和 OWASP 分類
- **修復建議**: 包含短期和長期修復建議
- **位置信息**: 精確定位漏洞位置（URL、參數、行號）

## 💻 如何使用

### 1. 快速開始

```python
# 使用統一掃描引擎
from services.scan.unified_scan_engine import UnifiedScanEngine

# 創建快速掃描
engine = UnifiedScanEngine.create_fast_scan(["https://example.com"])

# 執行掃描
results = await engine.run_comprehensive_scan()
print(f"掃描完成: {results['scan_id']}")
```

### 2. 使用 Python 掃描器

```python
from services.scan.aiva_scan import ScanOrchestrator
from ..aiva_common.schemas import ScanStartPayload

# 創建掃描請求
request = ScanStartPayload(
    scan_id="scan_001",
    targets=["https://example.com"],
    strategy="COMPREHENSIVE"
)

# 執行掃描
orchestrator = ScanOrchestrator()
result = await orchestrator.execute_scan(request)
```

### 3. 配置 TypeScript 引擎

```bash
# 安裝依賴
cd services/scan/aiva_scan_node
npm install

# 安裝瀏覽器
npm run install:browsers

# 啟動服務
npm run dev
```

### 4. 編譯 Rust 收集器

```bash
cd services/scan/info_gatherer_rust

# 開發模式
cargo run

# 釋出模式 (最佳化)
cargo build --release
.\target\release\info_gatherer_rust.exe
```

### 5. 工作程序模式

```python
# 啟動掃描工作程序 (透過 RabbitMQ)
from services.scan.aiva_scan.worker import run

# 監聽掃描任務
await run()
```

### 6. 生成 SARIF 報告

```python
from services.scan import SARIFConverter, Vulnerability

# 假設你已經有漏洞列表
vulnerabilities: list[Vulnerability] = [...]

# 轉換為 SARIF 格式
sarif_json = SARIFConverter.to_json(
    vulnerabilities=vulnerabilities,
    scan_id="scan_001"
)

# 保存為文件
with open("scan_results.sarif.json", "w") as f:
    f.write(sarif_json)

# 或者獲取 Python 對象
sarif_report = SARIFConverter.vulnerabilities_to_sarif(
    vulnerabilities=vulnerabilities,
    scan_id="scan_001"
)
```

## 📋 新增/刪減功能 SOP

### 新增功能 SOP

#### 1. 分析需求階段
```markdown
📝 需求分析檢查清單:
□ 確定功能屬於哪個引擎 (Python/TypeScript/Rust)
□ 評估性能要求和技術適用性
□ 檢查是否需要跨引擎整合
□ 確認 Schema 定義需求
```

#### 2. 技術選型指引

| 需求類型 | 推薦技術 | 理由 |
|---------|---------|------|
| 高性能文本處理 | Rust | 正則引擎性能最佳 |
| 瀏覽器自動化 | TypeScript + Playwright | 生態最成熟 |
| 複雜業務邏輯 | Python | 開發效率高 |
| 跨語言整合 | unified_scan_engine.py | 統一入口 |

#### 3. 開發流程

**Step 1: Schema 定義**
```python
# 1. 在 models.py 中添加新的資料模型
class NewFeatureConfig(BaseModel):
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

# 2. 更新 __init__.py 的 __all__ 列表
__all__ = [
    # ... 現有項目
    "NewFeatureConfig",
]
```

**Step 2: 核心實作**
```python
# Python 範例: 新增掃描器
class NewScanner:
    def __init__(self, config: NewFeatureConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    async def scan(self, target: str) -> List[Finding]:
        # 實作掃描邏輯
        pass
```

**Step 3: 整合到編排器**
```python
# 在 ScanOrchestrator 中整合
class ScanOrchestrator:
    def __init__(self):
        # ... 現有初始化
        self.new_scanner = NewScanner()
    
    async def execute_scan(self, request: ScanStartPayload):
        # ... 現有邏輯
        
        # 添加新功能
        if strategy_params.enable_new_feature:
            new_results = await self.new_scanner.scan(target)
            context.add_findings(new_results)
```

**Step 4: 測試驗證**
```python
# 創建單元測試
class TestNewScanner:
    async def test_basic_scan(self):
        scanner = NewScanner(NewFeatureConfig())
        results = await scanner.scan("https://example.com")
        assert len(results) >= 0
```

**Step 5: 文檔更新**
```markdown
# 更新本 README.md
## 新功能: XXX
- 功能描述
- 使用方式  
- 配置選項
```

### 刪減功能 SOP

#### 1. 影響評估
```markdown
🔍 刪減前檢查清單:
□ 檢查功能使用情況 (logs, metrics)
□ 確認依賴關係 (grep -r "功能名稱" services/)
□ 評估向後兼容性影響
□ 準備遷移計畫 (如果需要)
```

#### 2. 安全刪除流程

**Step 1: 標記為廢棄**
```python
import warnings

@deprecated("此功能將在 v2.0 移除，請使用 NewFeature 替代")
class OldScanner:
    def __init__(self):
        warnings.warn(
            "OldScanner 已廢棄，請遷移至 NewScanner",
            DeprecationWarning,
            stacklevel=2
        )
```

**Step 2: 配置開關**
```python
# 添加功能開關
class StrategyParameters:
    enable_old_feature: bool = False  # 預設關閉
```

**Step 3: 逐步移除**
```python
# 第一階段：條件執行
if strategy_params.enable_old_feature:
    # 舊功能代碼
    pass
else:
    logger.info("舊功能已停用")

# 第二階段：完全移除代碼
# (在確認無影響後)
```

**Step 4: 清理工作**
```bash
# 移除相關文件
rm services/scan/old_feature.py

# 更新導入
# 從 __init__.py 和其他文件移除相關導入

# 更新測試
# 移除相關測試案例

# 更新文檔
# 從 README.md 移除相關描述
```

### 版本控制最佳實踐

#### Git 工作流程
```bash
# 1. 創建功能分支
git checkout -b feature/new-scanner

# 2. 開發並提交
git add .
git commit -m "feat(scan): 添加新掃描器功能

- 實作 NewScanner 類別
- 整合到 ScanOrchestrator
- 添加單元測試
- 更新文檔

Closes #123"

# 3. 合併到主分支
git checkout main
git merge feature/new-scanner

# 4. 標記版本
git tag -a v1.2.0 -m "Release v1.2.0: 新增 NewScanner 功能"
```

#### 變更紀錄格式
```markdown
## [1.2.0] - 2025-10-24

### Added
- 新增 NewScanner 掃描器
- 支援 XXX 類型漏洞檢測
- 添加 Phase-I 高價值功能整合

### Changed  
- 改進 ScanOrchestrator 性能
- 更新統一掃描引擎配置介面

### Deprecated
- OldScanner 標記為廢棄，將在 v2.0 移除

### Removed
- 移除已廢棄的 LegacyFeature

### Fixed
- 修復 TypeScript 引擎記憶體洩漏問題
- 解決 Rust 收集器 Unicode 處理錯誤
```

### 監控與維護

#### 性能監控
```python
# 添加性能指標
import time
from prometheus_client import Counter, Histogram

SCAN_DURATION = Histogram('scan_duration_seconds', 'Scan duration')
SCAN_ERRORS = Counter('scan_errors_total', 'Scan errors')

class ScanOrchestrator:
    async def execute_scan(self, request):
        start_time = time.time()
        try:
            result = await self._do_scan(request)
            return result
        except Exception as e:
            SCAN_ERRORS.inc()
            raise
        finally:
            SCAN_DURATION.observe(time.time() - start_time)
```

#### 日誌管理
```python
# 結構化日誌
logger.info(
    "掃描完成",
    extra={
        "scan_id": request.scan_id,
        "targets_count": len(request.targets),
        "duration": context.scan_duration,
        "findings_count": len(context.findings)
    }
)
```

## � 問題處理標準流程

### 🎯 問題分類與處理原則

#### **P0 - 阻塞性問題** (24小時內解決)
- 掃描引擎無法啟動或崩潰
- 關鍵安全漏洞檢測失敗
- 多語言整合通信中斷

#### **P1 - 功能性問題** (3天內解決)  
- 特定目標掃描失敗
- 掃描結果不準確或遺漏
- 效能顯著下降

#### **P2 - 改善性問題** (1週內解決)
- UI/UX 改善需求
- 新功能請求
- 文檔更新需求

### 🔧 標準修復流程

1. **問題確認階段**
   ```bash
   # 1. 驗證問題重現
   python scripts/launcher/aiva_launcher.py --scan-test
   
   # 2. 收集系統狀態
   python testing/integration/aiva_module_status_checker.py
   
   # 3. 檢查相關日誌
   tail -f logs/scan_engine.log
   ```

2. **根因分析階段**
   - 使用調試工具定位問題
   - 檢查相關模組依賴
   - 驗證數據流完整性

3. **解決方案實施**
   - 編寫針對性修復代碼
   - 執行單元測試驗證
   - 進行集成測試確認

4. **結果驗證與文檔更新**
   - 確認問題完全解決
   - 更新相關文檔說明
   - 記錄修復過程供參考

---

## �🔧 故障排除

### 常見問題

#### 1. Python 引擎問題
```bash
# 依賴問題
pip install -r requirements.txt

# 導入錯誤
export PYTHONPATH="${PYTHONPATH}:/path/to/AIVA-git"

# 權限問題  
chmod +x scripts/launcher/aiva_launcher.py
```

#### 2. TypeScript 引擎問題
```bash
# Node.js 版本問題
nvm use 18

# 依賴安裝問題
rm -rf node_modules package-lock.json
npm install

# Playwright 瀏覽器問題
npx playwright install --with-deps
```

#### 3. Rust 引擎問題
```bash
# 編譯問題
cargo clean
cargo build --release

# 依賴問題
cargo update
```

#### 4. 整合問題
```bash
# RabbitMQ 連線問題
docker-compose up -d rabbitmq

# 權限問題
sudo chown -R $USER:$USER services/scan/

# 環境變數
export RABBITMQ_URL=amqp://aiva:dev_password@localhost:5672/
```

## 📊 性能基準

### 掃描性能對比

| 引擎類型 | 小型站點 (10 頁面) | 中型站點 (100 頁面) | 大型站點 (1000 頁面) |
|---------|-------------------|-------------------|-------------------|
| Python 核心 | ~30 秒 | ~5 分鐘 | ~45 分鐘 |
| TypeScript 動態 | ~45 秒 | ~8 分鐘 | ~60 分鐘 |
| Rust 資訊收集 | ~2 秒 | ~15 秒 | ~2 分鐘 |
| 統一引擎 | ~50 秒 | ~10 分鐘 | ~65 分鐘 |

### 資源消耗

| 引擎類型 | 記憶體消耗 | CPU 使用率 | 磁碟 I/O |
|---------|-----------|-----------|----------|
| Python | ~100 MB | ~20% | 中等 |
| TypeScript | ~200 MB | ~30% | 高 (瀏覽器快取) |
| Rust | ~5 MB | ~10% | 極低 |

## 🏆 最佳實踐

### 1. 掃描策略選擇
```python
# 快速掃描：適用於 CI/CD 流水線
UnifiedScanEngine.create_fast_scan(targets)

# 綜合掃描：適用於定期安全檢查  
UnifiedScanEngine.create_comprehensive_scan(targets)

# 自訂掃描：適用於特殊需求
config = UnifiedScanConfig(
    targets=targets,
    scan_type="comprehensive",
    max_depth=5,
    max_pages=500,
    enable_plugins=True
)
```

### 2. 效能優化
```python
# 平行掃描
strategy_params.requests_per_second = 10
strategy_params.connection_pool_size = 50

# 記憶體控制
strategy_params.max_pages = 100
strategy_params.enable_dynamic_scan = False  # 如不需要 JS 渲染
```

### 3. 安全配置
```python
# 認證配置
authentication = {
    "type": "bearer",
    "token": "your-api-token"
}

# 自訂標頭
custom_headers = {
    "User-Agent": "AIVA-Scanner/1.0",
    "X-Custom-Header": "value"
}
```

---

## 🔧 **開發規範與最佳實踐**

### 📐 **Scan 模組設計原則**

作為 AIVA 的統一掃描引擎,本模組必須在 Python/TypeScript/Rust 間維持一致的漏洞表示和評分標準。

#### 🎯 **使用 aiva_common 的核心原則**

**✅ Scan 模組的標準做法**（參考 `models.py` 正確實現）:

```python
# ✅ 正確 - Scan 模組的標準導入
from ..aiva_common.enums import (
    AssetType,               # 掃描目標類型
    Confidence,              # 檢測信心度
    ModuleName,              # 模組識別
    ScanProgress,            # 掃描進度狀態
    Severity,                # CVSS 嚴重程度映射
    VulnerabilityStatus,     # 漏洞生命週期狀態
)
from ..aiva_common.schemas import (
    CVEReference,            # CVE 標準引用
    CVSSv3Metrics,           # CVSS v3.x 評分
    CWEReference,            # CWE 分類
    SARIFResult,             # SARIF 2.1.0 格式
)

# ✅ 正確 - 使用標準化的 CVSS 結構
from services.scan.models import Vulnerability

vulnerability = Vulnerability(
    vuln_id="vuln_001",
    title="SQL Injection",
    description="SQL injection vulnerability found",
    severity=Severity.HIGH,
    confidence=Confidence.HIGH,
    vuln_type=VulnerabilityType.SQL_INJECTION,
    url="https://example.com/api/users",
    parameter="id",
    cvss_metrics=CVSSv3Metrics(
        base_score=8.5,
        attack_vector="NETWORK",
        attack_complexity="LOW",
        privileges_required="NONE",
        user_interaction="NONE",
        scope="UNCHANGED",
        confidentiality_impact="HIGH",
        integrity_impact="HIGH",
        availability_impact="NONE"
    ),
    cwe_ids=["CWE-89"],
    evidence=["Payload: ' OR 1=1--"]
)

# ✅ 正確 - 生成 SARIF 報告
from services.scan import SARIFConverter

sarif_report = SARIFConverter.vulnerabilities_to_sarif(
    vulnerabilities=[vulnerability],
    scan_id="scan_001"
)
```

#### 🆕 **新增或修改功能時的流程**

##### **⚙️ 執行前的準備工作 (必讀)**

**核心原則**: 充分利用現有資源，避免重複造輪子

在開始任何修改或新增掃描引擎功能前，務必執行以下檢查：

1. **檢查本機現有工具與插件**
   ```bash
   # 檢查專案內的輔助工具
   ls scripts/scan/                  # 查看 Scan 專用腳本
   ls tools/scan/                    # 查看掃描工具
   ls testing/scan/                  # 查看測試腳本
   
   # 常用工具和現有掃描引擎:
   # - services/scan/aiva_scan/vulnerability_scanner.py (漏洞掃描器)
   # - services/scan/aiva_scan/network_scanner.py (網路掃描)
   # - services/scan/aiva_scan/service_detector.py (服務探測)
   # - testing/scan/comprehensive_test.py (完整測試)
   # - testing/scan/juice_shop_real_attack_test.py (實戰測試)
   ```

2. **利用 VS Code 擴展功能**
   ```python
   # Pylance MCP 工具:
   # - pylanceFileSyntaxErrors: 檢查語法錯誤
   # - pylanceRunCodeSnippet: 測試掃描邏輯
   # - pylanceImports: 分析依賴關係
   
   # SonarQube 工具:
   # - sonarqube_analyze_file: 代碼質量檢查
   # - sonarqube_list_potential_security_issues: 安全問題檢測
   ```

3. **參考現有掃描引擎和工具**
   ```bash
   # 查看已實現的掃描功能
   ls services/scan/aiva_scan/*/
   
   # 參考專業掃描工具的實現:
   # Python 工具: ZAP (OWASP), Nuclei, Nikto
   # TypeScript 工具: Retire.js, ESLint Security
   # Rust 工具: Rustscan, Feroxbuster
   # Go 工具: Subfinder, Httpx, Katana
   ```

4. **功能不確定時，立即查詢最佳實踐**
   - 🌐 **掃描技術**: 查詢 OWASP Testing Guide, NIST 標準
   - 📚 **工具文檔**: 參考 Nmap, Masscan, ZAP 等工具的掃描技術
   - 🔍 **開源項目**: 使用 `github_repo` 搜索成熟的掃描引擎
   - 📊 **SARIF 標準**: 查詢 SARIF 2.1.0 規範文檔
   - 🛡️ **CVE 數據**: 使用 `fetch_webpage` 查詢 CVE 漏洞庫

5. **選擇最佳方案的判斷標準**
   - ✅ 優先使用 SARIF 2.1.0 標準輸出格式
   - ✅ 優先使用 CVSS v3.1 進行風險評分
   - ✅ 優先參考成熟工具（Nmap, ZAP, Nuclei）的掃描邏輯
   - ✅ 多語言引擎保持一致的數據格式（使用 aiva_common）
   - ⚠️ 避免自創掃描規則，參考 CWE/CAPEC 標準
   - ⚠️ 新掃描技術不確定時，先查詢業界實踐

**示例工作流程**:
```python
# 錯誤做法 ❌
# 直接開始寫掃描代碼，自己定義輸出格式

# 正確做法 ✅
# 步驟 1: 查找是否有類似掃描功能
ls services/scan/aiva_scan/
cat services/scan/aiva_scan/vulnerability_scanner.py  # 參考現有實現

# 步驟 2: 查詢專業工具的實現方式
# - Nmap 的端口掃描技術
# - ZAP 的主動/被動掃描
# - Nuclei 的模板引擎

# 步驟 3: 使用標準化格式
from aiva_common.schemas import SARIFResult, CVSSv3Metrics
from aiva_common.enums import Severity, Confidence

# 步驟 4: 參考 SARIF 規範
fetch_webpage("https://docs.oasis-open.org/sarif/sarif/v2.1.0/")

# 步驟 5: 使用工具檢查
pylance_analyze_file("new_scanner.py")
sonarqube_analyze_file("new_scanner.py")

# 步驟 6: 運行測試
python testing/scan/comprehensive_test.py
```

**多語言引擎開發參考**:
```python
# Python 引擎 - 參考工具
references_python = {
    "zap": "OWASP ZAP Python API",
    "nuclei": "Nuclei Template Engine",
    "nikto": "Nikto Web Scanner",
    "docs": "https://python-security.readthedocs.io/"
}

# TypeScript 引擎 - 參考工具
references_typescript = {
    "retire": "Retire.js (依賴漏洞掃描)",
    "eslint_security": "ESLint Security Plugin",
    "docs": "https://cheatsheetseries.owasp.org/cheatsheets/Nodejs_Security_Cheat_Sheet.html"
}

# Rust 引擎 - 參考工具
references_rust = {
    "rustscan": "高性能端口掃描",
    "feroxbuster": "Web 目錄爆破",
    "docs": "https://github.com/OWASP/crAPI (Rust 安全測試)"
}

# Go 引擎 - 參考工具
references_go = {
    "subfinder": "子域名發現",
    "httpx": "HTTP 探測",
    "katana": "網站爬蟲",
    "nuclei": "漏洞掃描",
    "docs": "https://github.com/projectdiscovery/"
}
```

---

##### **情境 1: 新增 Python 掃描引擎功能**

```python
# 步驟 1: 使用標準化的漏洞表示
from aiva_common.schemas import SARIFResult, CVSSv3Metrics
from aiva_common.enums import Severity, Confidence

def generate_vulnerability_report(finding: dict) -> SARIFResult:
    """生成符合 SARIF 標準的漏洞報告"""
    
    # ✅ 使用 CVSS 標準評分
    cvss = CVSSv3Metrics(
        base_score=7.5,
        attack_vector="NETWORK",
        attack_complexity="LOW",
        # ... CVSS 標準欄位
    )
    
    # ✅ 使用標準 Severity 映射
    severity = Severity.HIGH  # 基於 CVSS 分數
    
    return SARIFResult(
        ruleId=f"CWE-{finding['cwe_id']}",
        level="error",  # SARIF 標準: error/warning/note
        message=finding['description'],
        # ...
    )
```

##### **情境 2: 新增 TypeScript 掃描規則**

```typescript
// ✅ 正確 - TypeScript 中使用一致的枚舉值
export enum Severity {
    CRITICAL = "critical",  // 對應 Python Severity.CRITICAL
    HIGH = "high",
    MEDIUM = "medium",
    LOW = "low",
    INFO = "info"
}

// ✅ SARIF 標準結果格式
export interface SARIFResult {
    ruleId: string;
    level: "error" | "warning" | "note";  // SARIF 標準級別
    message: string;
    locations?: SARIFLocation[];
}

// ❌ 禁止 - 不要使用自定義嚴重程度
export enum CustomSeverity {
    VeryBad = "very_bad",   // ❌ 與標準不一致
    SoBad = "so_bad"        // ❌ 不符合 CVSS 映射
}
```

##### **情境 3: 新增 Rust 掃描模組**

```rust
// ✅ 正確 - Rust 中使用 CVSS 標準
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,  // CVSS 9.0-10.0
    High,      // CVSS 7.0-8.9
    Medium,    // CVSS 4.0-6.9
    Low,       // CVSS 0.1-3.9
    Info,      // CVSS 0.0
}

// ✅ CVSS v3 標準結構
#[derive(Serialize, Deserialize)]
pub struct CVSSv3Metrics {
    pub base_score: f32,
    pub attack_vector: String,      // NETWORK/ADJACENT/LOCAL/PHYSICAL
    pub attack_complexity: String,  // LOW/HIGH
    pub privileges_required: String, // NONE/LOW/HIGH
    // ... 其他 CVSS 標準欄位
}

// ❌ 禁止 - 不要使用非標準評分系統
pub struct CustomScore {
    pub danger_level: i32,  // ❌ 應使用 CVSS base_score
}
```

#### 🌐 **多語言掃描引擎協作**

**統一的漏洞報告格式**:

```python
# Python Engine（ZAP/Nuclei wrapper）
from aiva_common.schemas import SARIFResult

def wrap_zap_result(zap_alert: dict) -> SARIFResult:
    """將 ZAP 結果轉換為 SARIF 標準"""
    return SARIFResult(
        ruleId=f"ZAP-{zap_alert['pluginId']}",
        level=map_risk_to_sarif_level(zap_alert['risk']),
        message=zap_alert['alert'],
        locations=[{
            "physicalLocation": {
                "artifactLocation": {"uri": zap_alert['url']},
                "region": {"startLine": 1}
            }
        }]
    )
```

```typescript
// TypeScript Engine（自定義規則）
interface SARIFResult {
    ruleId: string;
    level: "error" | "warning" | "note";
    message: string;
    locations?: Array<{
        physicalLocation: {
            artifactLocation: { uri: string };
            region?: { startLine: number; endLine?: number };
        }
    }>;
}

function generateSARIFResult(
    vulnType: string,
    severity: Severity,
    location: string
): SARIFResult {
    return {
        ruleId: `TS-${vulnType}`,
        level: mapSeverityToSARIF(severity),
        message: `Detected ${vulnType} vulnerability`,
        locations: [/* ... */]
    };
}
```

```rust
// Rust Engine（高性能掃描）
use serde_json::json;

pub fn create_sarif_result(
    rule_id: &str,
    severity: Severity,
    message: &str,
    uri: &str
) -> serde_json::Value {
    json!({
        "ruleId": rule_id,
        "level": map_severity_to_sarif(&severity),
        "message": message,
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": uri},
                "region": {"startLine": 1}
            }
        }]
    })
}
```

#### � **修改現有功能的檢查清單**

在修改 Scan 模組任何掃描引擎前:

- [ ] **CVSS 合規**: 確保使用 CVSSv3Metrics 進行評分
- [ ] **SARIF 標準**: 輸出必須符合 SARIF 2.1.0 規範
- [ ] **CWE/CVE 引用**: 使用標準 ID 格式（CWE-79, CVE-2024-1234）
- [ ] **跨引擎一致性**: Python/TypeScript/Rust 產生的報告格式一致
- [ ] **性能測試**: 驗證新功能不影響掃描性能指標

#### 🧪 **Scan 模組特殊驗證**

```bash
# 1. 檢查是否有非標準 Severity 定義
grep -r "class.*Severity.*Enum" services/scan --include="*.py" --exclude-dir=__pycache__

# 2. 驗證 SARIF 輸出格式
python -m services.scan.validators.sarif_validator \
    --input test_output.sarif.json

# 3. 驗證 TypeScript 掃描引擎
cd services/scan/ts_engine
npm test -- --grep "SARIF compliance"

# 4. 驗證 Rust 掃描引擎
cd services/scan/rust_scanner
cargo test sarif_format_test

# 5. 跨引擎報告一致性測試
pytest services/scan/tests/integration/test_multi_engine_consistency.py -v

# 6. CVSS 評分驗證
python -c "
from services.scan.models import VulnerabilityReport
from ..aiva_common.schemas import CVSSv3Metrics
# 測試 CVSS 計算邏輯
"
```

#### 📊 **掃描結果標準化流程**

```mermaid
graph TD
    A[原始掃描結果] --> B{掃描引擎類型}
    B -->|Python/ZAP| C[ZAP Alert]
    B -->|TypeScript| D[Custom Finding]
    B -->|Rust| E[High-Perf Result]
    
    C --> F[轉換為 SARIF]
    D --> F
    E --> F
    
    F --> G[添加 CVSS 評分]
    G --> H[映射到標準 Severity]
    H --> I[添加 CWE/CVE 引用]
    I --> J[統一 SARIF 輸出]
    
    J --> K{驗證}
    K -->|通過| L[存入數據庫]
    K -->|失敗| M[記錄錯誤]
```

#### 🎯 **Scan 專屬的合理擴展**

```python
# ✅ 合理的 Scan 專屬枚舉（掃描技術細節）
class ScanTechnique(str, Enum):
    """掃描技術類型 - Scan 模組內部使用"""
    PASSIVE_CRAWL = "passive"           # 被動爬取
    ACTIVE_INJECTION = "active"         # 主動注入測試
    FUZZING = "fuzzing"                 # 模糊測試
    STATIC_ANALYSIS = "static"          # 靜態分析
    # 這些是掃描引擎的執行策略，不需要跨模組共享

class ScanEngineType(str, Enum):
    """掃描引擎類型識別"""
    ZAP = "zap"
    NUCLEI = "nuclei"
    CUSTOM_TS = "typescript"
    CUSTOM_RUST = "rust"
    # 這是掃描引擎的技術分類，用於路由和調度
```

#### 📝 **引擎特定注意事項**

**Python 開發者（ZAP/Nuclei wrapper）**:
- ✅ 將第三方工具結果轉換為 SARIF 標準
- ✅ 使用 CVSSv3Metrics 計算標準評分
- ❌ 不要保留工具特定的評分系統

**TypeScript 開發者（自定義規則）**:
- ✅ 實現符合 SARIF 的 TypeScript 接口
- ✅ 使用 `level: "error" | "warning" | "note"` 而非自定義級別
- ❌ 避免使用數字代碼代替語義化嚴重程度

**Rust 開發者（高性能掃描）**:
- ✅ 使用 `serde_json` 生成標準 SARIF JSON
- ✅ 確保序列化欄位名與 SARIF 規範一致
- ❌ 不要為了性能犧牲標準合規性

---

�📝 **文檔版本**: v1.1.0  
🔄 **最後更新**: 2025-10-25  
👥 **維護者**: AIVA Development Team  

有任何問題或建議，請提交 Issue 或 Pull Request！