# AIVA 功能模組開發指南

> **📋 適用對象**: 功能模組開發者、架構師、團隊負責人  
> **🎯 使用場景**: 功能模組實作開發、架構遵循、品質保證  
> **📅 建立日期**: 2025-11-06  
> **🔧 基礎要求**: 四語言開發環境 (Python + TypeScript + Rust + Go)

---

## 📑 目錄

1. [🎯 功能模組架構概覽](#-功能模組架構概覽)
2. [📊 模組完成度狀況](#-模組完成度狀況)
3. [🏗️ 標準四組件架構](#-標準四組件架構)
4. [🚨 急需實現模組指南](#-急需實現模組指南)
5. [⏳ 架構完善模組指南](#-架構完善模組指南)
6. [🔧 組件補強模組指南](#-組件補強模組指南)
7. [🌟 最佳實踐參考](#-最佳實踐參考)
8. [🔄 GO模組遷移指南](#-go模組遷移指南)
9. [🛠️ 開發環境配置](#-開發環境配置)
10. [📝 實施時間線](#-實施時間線)

---

## 🎯 功能模組架構概覽

### AIVA v5 五大模組架構

```
AIVA v5 架構
├── 🧠 Core (23 AI components)
│   └── AI驅動的核心決策引擎
├── ⚙️ Features (10 security functions)  ← 📍 本指南重點
│   ├── XSS (4/4) ✅
│   ├── SQLI (3/4) ⏳
│   ├── AUTHN_GO (2/4) ⏳
│   ├── CRYPTO (0/4) 🚨
│   ├── POSTEX (0/4) 🚨
│   ├── IDOR (0/4) 🔧
│   └── SSRF (0/4) 🔧
├── 🔗 Integration (12 enterprise components)
├── 🔍 Scan (15 scanning components + 3 GO modules)
│   ├── SSRF_GO (遷移中) 🔄
│   ├── CSPM_GO (遷移中) 🔄
│   └── SCA_GO (遷移中) 🔄
└── 📚 aiva_common (unified standards)
```

---

## 📊 模組完成度狀況

### 完成度矩陣 (2025-11-06)

| 模組名稱 | Worker | Detector | Engine | Config | 完成度 | 優先級 | 預估工作量 |
|----------|--------|----------|--------|--------|--------|--------|-----------|
| **XSS** | ✅ | ✅ | ✅ | ✅ | **4/4** | 🌟 參考標準 | 0週 (完成) |
| **SQLI** | ✅ | ✅ | ✅ | ⏳ | **3/4** | 🔥 高優先級 | 1週 |
| **AUTHN_GO** | ✅ | ✅ | ⏳ | ⏳ | **2/4** | 🔥 高優先級 | 2週 |
| **CRYPTO** | ⏳ | ⏳ | ⏳ | ⏳ | **0/4** | 🚨 **緊急** | 4週 |
| **POSTEX** | ⏳ | ⏳ | ⏳ | ⏳ | **0/4** | 🚨 **緊急** | 4週 |
| **IDOR** | ⏳ | ⏳ | ⏳ | ⏳ | **0/4** | 🔧 標準優先級 | 3週 |
| **SSRF** | ⏳ | ⏳ | ⏳ | ⏳ | **0/4** | 🔧 標準優先級 | 3週 |

### GO模組架構遷移狀況

| GO模組 | 當前位置 | 目標位置 | 遷移原因 | 遷移時程 |
|--------|----------|----------|----------|----------|
| **SSRF_GO** | Features | **Scan** | 適合廣域掃描 | 4週內 |
| **CSPM_GO** | Features | **Scan** | 雲端資源掃描 | 4週內 |
| **SCA_GO** | Features | **Scan** | 軟體組件分析 | 4週內 |
| **AUTHN_GO** | Features | **Features** | 深度認證測試 | 保持不變 |

---

## 🏗️ 標準四組件架構

每個功能模組都應遵循統一的四組件架構模式：

### 📐 組件定義

```python
# 標準四組件架構範例 (以XSS為範本)
🔧 Worker      # 工作執行器 - 實際執行安全檢測任務
🔍 Detector    # 檢測器 - 識別和分析安全漏洞
⚙️ Engine      # 引擎 - 核心處理邏輯和決策引擎
📋 Config      # 配置 - 模組配置和參數管理
```

### 🎯 架構範本 (參考XSS模組)

```
services/features/xss/
├── worker/
│   ├── xss_worker.py        # 主要工作執行器
│   └── __init__.py
├── detector/
│   ├── xss_detector.py      # 漏洞檢測核心
│   └── __init__.py
├── engine/
│   ├── xss_engine.py        # 處理引擎
│   └── __init__.py
├── config/
│   ├── xss_config.py        # 配置管理
│   └── __init__.py
├── __init__.py
└── README.md                # 模組文檔
```

---

## 🚨 急需實現模組指南

> **目標模組**: CRYPTO + POSTEX  
> **完成度**: 0/4 (完全空白)  
> **緊急程度**: 🚨 **最高優先級**  
> **預估工作量**: 8週 (4週 × 2模組)

### 📋 CRYPTO 模組實作計劃

#### Week 1-2: 基礎架構建立
```bash
# 1. 創建模組目錄結構
mkdir -p services/features/crypto/{worker,detector,engine,config}

# 2. 基礎組件實現 (參照XSS範本)
- crypto_worker.py     # 加密漏洞檢測執行器
- crypto_detector.py   # 弱加密算法檢測器
- crypto_engine.py     # 加密分析引擎
- crypto_config.py     # 加密檢測配置
```

#### Week 3-4: 功能實現與測試
- **加密算法弱點檢測**: MD5, SHA1, DES等不安全算法
- **密鑰管理檢測**: 硬編碼密鑰、弱密鑰生成
- **SSL/TLS漏洞檢測**: 協議版本、憑證驗證
- **隨機數生成檢測**: 可預測的隨機數使用

### 📋 POSTEX 模組實作計劃

#### Week 1-2: 基礎架構建立
```bash
# 1. 創建模組目錄結構
mkdir -p services/features/postex/{worker,detector,engine,config}

# 2. 基礎組件實現
- postex_worker.py     # 後滲透檢測執行器
- postex_detector.py   # 權限提升檢測器
- postex_engine.py     # 後滲透分析引擎
- postex_config.py     # 後滲透檢測配置
```

#### Week 3-4: 功能實現與測試
- **權限提升檢測**: Sudo漏洞、SUID程序檢測
- **橫向移動檢測**: 內網滲透路徑分析
- **持久化檢測**: 後門、定時任務檢測
- **數據收集檢測**: 敏感文件、憑證收集

---

## ⏳ 架構完善模組指南

> **目標模組**: SQLI (3/4) + AUTHN_GO (2/4)  
> **完成度**: 部分完成，需補強  
> **優先程度**: 🔥 高優先級  
> **預估工作量**: 3週

### 📋 SQLI 模組補強計劃 (缺Config組件)

#### Week 1: Config組件實現
```python
# services/features/sqli/config/sqli_config.py
class SQLIConfig:
    """SQL注入檢測配置管理器"""
    
    # 檢測規則配置
    PAYLOADS = {
        'basic': ["' OR '1'='1", "'; DROP TABLE--"],
        'union': ["UNION SELECT NULL--"],
        'blind': ["' AND (SELECT SUBSTRING(@@version,1,1))='5'--"]
    }
    
    # 資料庫類型配置
    DB_SIGNATURES = {
        'mysql': ['mysql', 'version()'],
        'postgresql': ['postgresql', 'version()'],
        'oracle': ['oracle', 'dual']
    }
```

### 📋 AUTHN_GO 模組補強計劃 (缺Engine+Config)

#### Week 1: Engine組件實現 (Go)
```go
// services/features/authn_go/engine/authn_engine.go
package engine

type AuthnEngine struct {
    config *Config
    client *http.Client
}

func (e *AuthnEngine) AnalyzeAuth(target string) (*AuthResult, error) {
    // JWT token 分析
    // 認證繞過檢測
    // 會話管理檢測
}
```

#### Week 2: Config組件實現 (Go)
```go
// services/features/authn_go/config/authn_config.go
package config

type AuthnConfig struct {
    JWTSecrets     []string `json:"jwt_secrets"`
    WeakPasswords  []string `json:"weak_passwords"`
    SessionTimeout int      `json:"session_timeout"`
}
```

---

## 🔧 組件補強模組指南

> **目標模組**: IDOR (0/4) + SSRF (0/4)  
> **完成度**: 完全空白  
> **優先程度**: 🔧 標準優先級  
> **預估工作量**: 6週 (3週 × 2模組)

### 📋 IDOR 模組實作計劃

#### Week 1: 架構建立
- 創建標準四組件目錄結構
- 參考XSS模組範本進行基礎實現

#### Week 2-3: 功能實現
- **直接對象引用檢測**: ID遍歷檢測
- **授權檢測**: 水平/垂直權限檢測
- **參數污染檢測**: HTTP參數篡改檢測

### 📋 SSRF 模組實作計劃

#### Week 1: 架構建立
- 創建標準四組件目錄結構
- 規劃與SSRF_GO遷移協調

#### Week 2-3: 功能實現
- **內網資源檢測**: 私有IP掃描
- **雲端元數據檢測**: AWS/Azure/GCP元數據訪問
- **協議檢測**: HTTP/FTP/FILE協議濫用

---

## 🌟 最佳實踐參考

### XSS模組架構分析 (4/4完成)

```python
# 參考實現路徑: services/features/xss/
📊 統計資料:
├── 檔案數量: 10 files
├── 代碼行數: 2,511 lines  
├── 四組件: Worker ✅ + Detector ✅ + Engine ✅ + Config ✅
└── 完成度: 100% (標準範本)

🎯 核心特點:
├── 統一錯誤處理機制
├── 標準化配置管理
├── 完整的日誌記錄
├── 全面的測試覆蓋
└── 清晰的API設計
```

### 開發規範要點

1. **命名規範**: `{module_name}_{component_type}.py`
2. **導入規範**: 統一使用相對導入
3. **錯誤處理**: 實現標準異常類別
4. **配置管理**: 支援環境變數和配置文件
5. **日誌記錄**: 使用統一日誌格式
6. **測試覆蓋**: 每個組件都要有對應測試

---

## 🔄 GO模組遷移指南

### 遷移架構設計

```
services/scan/go_scanners/          # 新的GO模組統一位置
├── ssrf_scanner/                   # SSRF_GO → ssrf_scanner
│   ├── worker/
│   ├── detector/
│   ├── engine/
│   └── config/
├── cspm_scanner/                   # CSPM_GO → cspm_scanner
├── sca_scanner/                    # SCA_GO → sca_scanner
├── shared/                         # 共用組件
│   ├── amqp_client.go             # AMQP通訊客戶端
│   ├── sarif_formatter.go         # SARIF結果格式化
│   └── base_scanner.go            # 基礎掃描器接口
└── README.md
```

### 遷移實施步驟

#### Phase 1: 準備工作 (Week 1)
1. **建立新目錄結構**: `services/scan/go_scanners/`
2. **設計統一接口**: AMQP通訊 + SARIF格式
3. **創建基礎組件**: 共用掃描器基類

#### Phase 2: 模組遷移 (Week 2-3)
1. **SSRF_GO遷移**: 保持原有功能，適配新架構
2. **CSPM_GO遷移**: 雲端安全掃描功能整合
3. **SCA_GO遷移**: 軟體組件分析功能整合

#### Phase 3: 整合測試 (Week 4)
1. **跨模組通訊測試**: AMQP訊息佇列測試
2. **SARIF結果驗證**: 統一結果格式驗證
3. **性能基準測試**: 並發掃描性能測試

---

## 🛠️ 開發環境配置

### 必要工具安裝

```bash
# Python 環境 (3.11+)
python --version  # 確認版本

# Node.js 環境 (18+)
node --version
npm --version

# Rust 環境 (1.70+)
rustc --version
cargo --version

# Go 環境 (1.21+)
go version

# 開發工具
pip install black flake8 mypy pytest
npm install -g typescript ts-node
cargo install rustfmt clippy
go install golang.org/x/tools/cmd/goimports@latest
```

### VS Code 配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "rust-analyzer.check.command": "clippy",
    "go.formatTool": "goimports",
    "typescript.preferences.importModuleSpecifier": "relative"
}
```

### Docker 開發環境

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  aiva-dev:
    image: aiva:dev
    volumes:
      - .:/workspace
    environment:
      - PYTHONPATH=/workspace
      - GO111MODULE=on
    ports:
      - "8080:8080"    # API服務
      - "5672:5672"    # RabbitMQ
```

---

## 📝 實施時間線

### 整體時間規劃 (15週)

```
Phase 1: 緊急實現 (Week 1-8)
├── Week 1-4: CRYPTO 模組完整實現
├── Week 4-8: POSTEX 模組完整實現
└── 並行: GO模組遷移準備

Phase 2: 架構完善 (Week 9-11)
├── Week 9: SQLI Config組件補強
├── Week 10-11: AUTHN_GO Engine+Config實現
└── 並行: GO模組遷移實施

Phase 3: 組件補強 (Week 12-15)
├── Week 12-13: IDOR 四組件完整實現
├── Week 14-15: SSRF 四組件完整實現
└── 整合測試與性能優化
```

### 里程碑檢查點

- **Week 4**: CRYPTO模組基礎功能驗證
- **Week 8**: POSTEX模組基礎功能驗證  
- **Week 11**: SQLI+AUTHN_GO補強完成
- **Week 15**: 所有功能模組完成，系統整合測試

### 品質保證要求

1. **代碼審查**: 每個組件完成後進行同伴審查
2. **單元測試**: 測試覆蓋率不低於80%
3. **整合測試**: 跨模組通訊功能驗證
4. **性能測試**: 並發處理能力驗證
5. **安全檢測**: 使用SAST工具進行安全掃描

---

## 🔗 相關資源

### 📋 技術報告參考
- [功能模組完成總結](../../FEATURE_MODULES_COMPLETION_SUMMARY.md)
- [01_CRYPTO_POSTEX_急需實現報告](../../reports/features_modules/01_CRYPTO_POSTEX_急需實現報告.md)
- [02_SQLI_AUTHN_GO_架構完善報告](../../reports/features_modules/02_SQLI_AUTHN_GO_架構完善報告.md)
- [06_XSS_最佳實踐架構參考報告](../../reports/features_modules/06_XSS_最佳實踐架構參考報告.md)

### 🛠️ 開發工具指南
- [Python開發指南](./PYTHON_DEVELOPMENT_GUIDE.md)
- [Go開發指南](./GO_DEVELOPMENT_GUIDE.md)
- [Rust開發指南](./RUST_DEVELOPMENT_GUIDE.md)
- [開發環境快速設置](../development/DEVELOPMENT_QUICK_START_GUIDE.md)

### 🏗️ 架構文檔
- [模組遷移指南](./MODULE_MIGRATION_GUIDE.md)
- [跨語言Schema指南](../architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md)
- [Docker部署指南](../deployment/DOCKER_GUIDE.md)

---

**📝 文檔資訊**
- **維護者**: AIVA 功能模組開發團隊
- **建立日期**: 2025-11-06
- **更新頻率**: 隨開發進度即時更新
- **版本**: v1.0 (配合功能模組需求文件完成)