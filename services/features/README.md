# AIVA Features 模組

> **版本**: v7.1 | **狀態**: ✅ 架構整合完成 | **更新**: 2026-01-09

## 🔔 最新更新 (2026-01-09)

### ✅ 重大進展：AI Commander 整合完成

1. **AttackCoordinator 整合** - [`attack_coordinator.py`](../core/aiva_core/task_planning/commander/attack_coordinator.py)
   - ✅ 直接調用 Features 功能模組 (sqli/xss/ssrf/idor)
   - ✅ 整合 AttackExecutor 和 MultiEngineCoordinator
   - ✅ Phase 2 攻擊流程完整可用

2. **決策引擎連動** - [`enhanced_decision_agent.py`](../core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py)
   - ✅ `decide_phase2_targets()` 返回攻擊目標優先級
   - ✅ AI 決策結果直接驅動功能模組執行

### 🔔 歷史更新 (2025-12-20)

### ✅ 重大架構變更

1. **新增功能模組** - [`function_info_leak`](features_ready/function_info_leak/README.md)
   - ✅ 從 Scan 模組移入敏感信息檢測器（547 lines）
   - ✅ 支持 API 密鑰、JWT、憑證、信用卡等檢測
   - ✅ 完整的 README 和使用文檔

2. **明確模組職責**
   - ✅ Features：所有 Python 檢測邏輯
   - ✅ Scan：僅負責多語言引擎調度（Go/Rust/TypeScript）
   - ✅ 刪除重複組件（HTML 解析、指紋識別等）

3. **功能模組完整性確認**
   - ✅ XSS：使用外部工具（XSStrike/Dalfox）自帶解析器
   - ✅ SQLI：直接接收測試目標，不需要爬蟲
   - ✅ IDOR：自己提取 ID，不需要 HTML 解析
   - ✅ SSRF：直接測試 URL，不需要爬蟲

### 📝 架構原則

**功能模組設計原則**：
- ✅ 每個模組只包含自己需要的工具
- ✅ 不追求"共享基礎設施"的過度抽象
- ✅ 外部工具優先（XSStrike > 自己寫解析器）
- ✅ 模組間不共享代碼（避免耦合）

---

**快速導航**: [← Services](../README.md) | [📐 架構設計](./SIMPLE_ARCHITECTURE.md) | [🎯 Scan 模組](../scan/README.md)

---

## 📚 目錄

1. [📐 架構設計](#-架構設計)
2. [🎯 現況與整合計劃](#-現況與整合計劃)
3. [📊 功能模組狀態](#-功能模組狀態)
4. [🔗 功能模組導航](#-功能模組導航)
5. [🛠️ 開發指南](#️-開發指南)
6. [📋 快速參考](#-快速參考)

---

## 📐 架構設計

- **[SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md)** - ⭐ **最新簡化架構設計 (v5.0)**
  - AI Commander 直接調用功能模組
  - 功能模組提供同步接口：`scan(target, options) -> dict`
  - 無需適配器層，無需命令處理器
  - **核心理念**: 「功能決定架構，不同功能不同架構」

- **[ARCHITECTURE_CORRECTION.md](./ARCHITECTURE_CORRECTION.md)** - 🔧 **架構修正說明**
  - 為何不需要統一管理器模式
  - 為何不需要強制 Python 回退
  - 為何不需要統一 `__init__.py` 導出
  - 每個模組應該按功能特性獨立設計
  
- **[ARCHITECTURE_ANALYSIS_AND_CORRECTION.md](./ARCHITECTURE_ANALYSIS_AND_CORRECTION.md)** - 架構演進歷史

---

## 🎯 現況與整合狀態

### 📌 當前狀態 (2026-01-09 更新)

**功能模組**：✅ 已有完整檢測邏輯（同步實現）

**AI Commander 整合**：✅ **已完成** - AttackCoordinator 直接調用功能模組

```python
# services/features/function_xss/integration_tools/xss_tools.py
class XSSManager:
    def comprehensive_scan(self, target_url: str, options: Dict) -> dict:
        # 真實的 XSS 檢測代碼
        return {"vulnerable": True, "payloads": [...]}
```

**AI Commander**：✅ **已整合** (2026-01-09)

```python
# services/core/aiva_core/task_planning/commander/attack_coordinator.py
# 實際整合代碼 - AttackCoordinator 直接調用功能模組

class AttackCoordinator:
    async def detect_vulnerabilities(self, context: dict) -> dict:
        """調用功能模組檢測漏洞"""
        module_map = {
            "sqli": "services.features.function_sqli.worker",
            "xss": "services.features.function_xss.worker",
            "ssrf": "services.features.function_ssrf.worker",
            "idor": "services.features.function_idor.worker",
        }
        # 動態導入並執行對應模組
        ...
```

### 🎯 整合方式

按 [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md) 設計：

- **外部工具**：AI Commander 通過 `subprocess` 調用（如 sqlmap、XSStrike）
- **AIVA 模組**：AI Commander 通過 `import + asyncio.to_thread()` 調用
- **功能模組**：提供簡單同步函數，返回字典結果

---

## 📊 功能模組狀態

> **完整分析**: 2025-12-11  
> **架構**: 同步函數提供 `scan(target, options) -> dict` 接口  
> **範圍**: 不包含需人工操作的模組（social_engineering、forensic 等）  
> **⚠️ 重組通知**: 2025-12-17 已按完成度重組目錄結構，詳見 [_FEATURES_REORGANIZATION.md](_FEATURES_REORGANIZATION.md)

### ✅ 高完成度 - 可直接整合 (7個) → `features_ready/`

| 模組 | 完成度 | 整合狀態 | 核心功能 | 文檔連結 |
|------|--------|---------|---------|----------|
| **function_sqli** | 95% | ✅ v2.1.0 | 6種檢測引擎、CommandHandler | [README](features_ready/function_sqli/README.md) |
| **function_xss** | 90% | ✅ v2.1.0 | 4種檢測器、外部工具整合 | [README](features_ready/function_xss/README.md) |
| **function_ssrf** | 85% | ✅ v2.0.0 | 內網探測、OAST技術 | [README](features_ready/function_ssrf/README.md) |
| **function_idor** | 80% | ✅ v1.0.0 | 權限矩陣、資源ID提取 | [README](features_ready/function_idor/README.md) |
| **function_bizlogic** | 70% | ✅ v1.1.0 | 價格操縱/競態/工作流繞過 | [README](features_ready/function_bizlogic/README.md) |
| **function_crypto** | 50% | ⏳ v2.0.0 | Rust CLI（需編譯） | [README](features_ready/function_crypto/README.md) |
| **function_info_leak** | 100% | ✅ v1.0.0 | ⭐ 敏感信息檢測（API密鑰/JWT）| [README](features_ready/function_info_leak/README.md) |

**特徵**：
- ✅ 完整的檢測邏輯實現
- ✅ 有工作器（worker.py）或統一接口
- ✅ **2025-12-20 新增**: function_info_leak 從 Scan 模組移入（547 lines）
- ✅ 所有模組功能完整，無重複代碼

### ⚠️ 中等完成度 - 需完善檢測邏輯 (3個) → `features_in_development/`

| 模組 | 完成度 | 整合狀態 | 現有架構 | 更新日期 |
|------|--------|---------|---------|---------|
| **[function_authn_go](features_in_development/function_authn_go/README.md)** | 50% | ⏳ v1.2.0 | authn_wrapper.py已完成、BUILD_GUIDE.md已創建 | 2025-12-17 |
| **[function_postex](features_in_development/function_postex/README.md)** | 50% | ⏳ v1.2.0 | PostExManager架構完成、需增強檢測邏輯 | 2025-12-17 |
| **[function_web_scanner](features_in_development/function_web_scanner/README.md)** | 35% | ⏳ v1.2.0 | WebScannerManager架構完成、缺少README | 2025-12-17 |

**下一步**:
- ✅ function_authn_go: **編譯 Go 引擎**（參見 BUILD_GUIDE.md）
- ⏳ function_postex: 完善 lateral_movement、persistence、privilege_escalation 檢測邏輯
- ⏳ function_web_scanner: 創建 README.md、完善檢測邏輯

**✨ 核心理念**（依照 [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md)）：

> **「功能決定架構，不同功能不同架構」**
> 
> **「有問題就修復，不繞過問題」**

**唯一的統一要求**:
- ✅ 提供簡單的 `scan()` 類型接口
- ✅ 返回清晰的數據結構
- ✅ 錯誤處理明確

**不需要統一**:
- ❌ 不強制統一的類別名稱
- ❌ 不強制統一的文件結構
- ❌ 不需要 Python 回退實現
- ✅ 每個模組用最適合的技術實現

**重要原則**:
- 🔴 **Go 模組就用 Go** - 不需要 Python 回退
- 🔴 **Rust 模組就用 Rust** - 不需要 Python 回退  
- 🔴 **未編譯就報錯** - 提供明確的編譯指引
- ✅ 有編譯問題就修復，不降級到 Python

**架構說明**（每個模組按功能特性獨立設計）：

```python
# 功能模組 - 簡單同步實現（各自按最佳實踐組織）

# function_sqli - 多引擎架構
from services.features.function_sqli.detector import SQLInjectionDetector
detector = SQLInjectionDetector()
result = detector.scan(target)  # 使用多個測試引擎

# function_xss - DOM 分析架構
from services.features.function_xss.integration_tools.xss_tools import XSSManager
manager = XSSManager()
result = manager.comprehensive_scan(target, options)  # Context-aware 分析

# function_authn_go - Go 語言實現
# 可能直接調用 Go 二進制文件或使用 subprocess

# AI Commander - 統一的異步調度
class SecurityScanner:
    async def scan_sqli(self, target: str):
        detector = SQLInjectionDetector()
        # 用 asyncio.to_thread() 包裝任何同步調用
        result = await asyncio.to_thread(detector.scan, target)
        return result
```

**關鍵點**：
- ✅ 每個模組使用最適合的架構（SQLi 用引擎模式，XSS 用 Manager 模式等）
- ✅ 只要提供同步的 `scan()` 接口即可
- ✅ AI Commander 統一用 `asyncio.to_thread()` 處理異步
- ❌ 不強制所有模組使用相同的類別名稱或結構

### 🗑️ 低完成度 - 已歸檔 (2個) → `features_archived/`

| 模組 | 完成度 | 狀態 | 說明 |
|------|--------|-----|------|
| **[function_exploit_framework](features_archived/function_exploit_framework/README.md)** | 25% | 🟡 輔助 | 僅作為 PoC 驗證工具，非主力掃描 |
| **[function_payload_generator](features_archived/function_payload_generator/)** | - | 🟡 輔助 | 輔助工具，可整合到其他模組 |

**特徵**：
- ⚠️ 功能敏感，需額外授權
- 🎯 定位為輔助增強工具（獎金翻倍器）
- 📝 不應作為主動掃描工具
- 💾 保留備用，定期評估是否恢復開發

### 🗑️ 已完全移除模組 (1個)

| 模組 | 原因 | 備份位置 |
|------|------|---------|
| **function_ddos** | 不適用於 Bug Bounty，法律風險極高 | `新增資料夾 (3)/function_ddos_archived/` |

**說明**：
- ❌ 所有主流 Bug Bounty 平台都禁止傳統 DDoS 測試
- ⚠️ 法律風險過高（可能構成犯罪）
- 📋 詳見 [BOUNTY_EARNING_ANALYSIS.md](BOUNTY_EARNING_ANALYSIS.md)

### 🚫 需人工操作 - 暫不處理 (5個) → `features_manual_operation/`

| 模組 | 說明 | 不處理原因 |
|------|-----|-----------|
| **[function_social_engineering](features_manual_operation/function_social_engineering/README.md)** | 社會工程測試 | 需要OSINT收集、人員分析等複雜人工操作 |
| **[function_forensic](features_manual_operation/function_forensic/README.md)** | 數位鑑識 | 需要證據收集、人工分析判斷 |
| **[function_reverse_engineering](features_manual_operation/function_reverse_engineering/README.md)** | 逆向工程 | 需要人工分析二進制文件 |
| **[function_steganography](features_manual_operation/function_steganography/README.md)** | 隱寫術分析 | 需要人工判斷和分析 |
| **[function_wordlist_generator](features_manual_operation/function_wordlist_generator/README.md)** | 字典生成 | 需要人工定義生成策略 |

---

### 🎯 整合優先級

**Phase 1 - 立即整合** (2-3 週) - `features_ready/`：
1. function_sqli - 6種引擎，95%完成，✅ README 已完整
2. function_crypto - 純 Rust CLI（v2.0），95%完成 ✅
3. function_xss - 多種類型，90%完成
4. function_ssrf - OAST技術，85%完成
5. function_idor - 權限檢測，80%完成

**Phase 2 - 短期完善** (2-4 週) - `features_in_development/`：
6. function_bizlogic - 業務邏輯測試（已在 features_ready，需整合）
7. **function_authn_go** - 🔴 **必須編譯 Go 引擎** → 不提供 Python 回退
8. function_postex - 完善引擎檢測邏輯（使用 PostExDetector）
9. function_web_scanner - 完善檢測邏輯（使用 WebAttackManager）

**已歸檔** - `features_archived/`：exploit_framework（僅作輔助工具）、payload_generator（輔助工具）
**已移除**：function_ddos（不適用於 Bug Bounty，已移至備份資料夾）
**暫不處理** - `features_manual_operation/`：5個需人工操作的模組

### 📋 整合檢查清單

針對每個模組：

- [ ] **代碼檢查**
  - [ ] 確認完整檢測邏輯
  - [ ] 確認測試數據存在
  
- [ ] **接口標準化** (依照 [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md))
  - [ ] 提供 `scan()` 類型的接口（函數名可以是 scan, comprehensive_scan, detect 等）
  - [ ] 確保純同步實現（不使用 async/await）
  - [ ] 返回字典結果
  - [ ] 錯誤處理清晰
  - [ ] 由 AI Commander 負責異步調度（`asyncio.to_thread()`）
  - ❌ 不強制統一類別名稱（如 Manager）
  - ❌ 不強制統一導出方式

- [ ] **AI Commander 整合**
  - [ ] 在 SecurityScanner 添加調用方法
  - [ ] 使用 `asyncio.to_thread()` 包裝
  - [ ] 實現結果聚合

- [ ] **測試驗證**
  - [ ] 單元測試通過
  - [ ] 集成測試通過
  - [ ] 靶場實測

---

## 🔗 功能模組導航

> 💡 **重組說明**: 2025-12-17 已按完成度重組目錄結構，詳見 [_FEATURES_REORGANIZATION.md](_FEATURES_REORGANIZATION.md)

### 📚 核心功能模組

#### ✅ 高完成度 (6個) - `features_ready/`

- ⚡️ **[SQL注入檢測](features_ready/function_sqli/README.md)** - 6種引擎、多資料庫支援
- 🔒 **[密碼學檢測](features_ready/function_crypto/README.md)** - 純 Rust CLI（v2.0）、TLS/Cookie/Headers/JS 分析
- 🎭 **[XSS檢測](features_ready/function_xss/README.md)** - 反射型/存儲型/DOM XSS、Blind XSS
- 🌐 **[SSRF檢測](features_ready/function_ssrf/README.md)** - 內網探測、OAST技術、語義分析
- 🔐 **[IDOR檢測](features_ready/function_idor/README.md)** - 權限矩陣、資源ID提取
- 💼 **[業務邏輯檢測](features_ready/function_bizlogic/README.md)** - 價格操控、競態條件、流程繞過

#### ⚠️ 中等完成度 (3個) - `features_in_development/`

- 🔑 **[認證檢測](features_in_development/function_authn_go/README.md)** - Go高性能認證繞過檢測
- ⚡ **[後滲透](features_in_development/function_postex/README.md)** - 橫向移動、持久化引擎
- 🌐 **[Web掃描器](features_in_development/function_web_scanner/README.md)** - 綜合Web漏洞掃描

#### 🚫 需人工操作 (5個) - `features_manual_operation/`

- 🎭 **[社會工程](features_manual_operation/function_social_engineering/README.md)**
- 🔍 **[數位鑑識](features_manual_operation/function_forensic/README.md)**
- 🔧 **[逆向工程](features_manual_operation/function_reverse_engineering/README.md)**
- 🖼️ **[隱寫術分析](features_manual_operation/function_steganography/README.md)**
- 📝 **[字典生成](features_manual_operation/function_wordlist_generator/README.md)**

#### 🗑️ 已歸檔 (2個) - `features_archived/`

- 💣 **[漏洞利用框架](features_archived/function_exploit_framework/README.md)**
- 🎯 **[Payload生成器](features_archived/function_payload_generator/)**

#### 🛠️ 支援組件

- 💼 **[共用組件](common/README.md)** - Go/Python跨語言共用功能
- 🏗️ **[基礎組件](base/README.md)** - 功能基類、HTTP客戶端

### 📊 模組完成度統計

| 模組 | 語言 | 完成度 | 目錄 | 主要功能 |
|------|------|--------|------|---------|
| function_sqli | Python | 95% | features_ready | SQL注入檢測（6種引擎） |
| function_crypto | Rust | 95% | features_ready | 密碼學配置掃描（純CLI v2.0） |
| function_xss | Python | 90% | features_ready | XSS檢測（多種類型） |
| function_ssrf | Python | 85% | features_ready | SSRF檢測（OAST） |
| function_idor | Python | 80% | features_ready | 權限繞過檢測 |
| function_bizlogic | Python | 75% | features_ready | 業務邏輯測試 |
| function_authn_go | Go | 70% | features_in_development | 認證繞過（Go） |
| function_postex | Python | 50% | features_in_development | 後滲透利用 |
| function_web_scanner | Python | 35% | features_in_development | Web漏洞掃描 |

---

## 🛠️ 開發指南

### 🎯 功能模組設計原則

**核心理念**：簡單直接，AI Commander 直接調用

```python
# ✅ 功能模組標準實現
def scan(target: str, options: dict = None) -> dict:
    """
    同步掃描函數
    
    Args:
        target: 掃描目標（URL、IP等）
        options: 掃描選項
        
    Returns:
        dict: 掃描結果 {"vulnerable": bool, "details": ...}
    """
    # 實現檢測邏輯
    return result
```

**禁止的做法**：
- ❌ 使用 async/await（AI Commander 處理異步）
- ❌ 創建命令處理器（不需要）
- ❌ 添加適配器層（不需要）

### 📐 使用 aiva_common 標準

```python
# ✅ 正確 - 使用標準枚舉
from ..aiva_common.enums import (
    Severity,           # 嚴重程度
    Confidence,         # 信心度
    VulnerabilityStatus # 漏洞狀態
)
from ..aiva_common.schemas import (
    CVEReference,       # CVE引用
    CWEReference,       # CWE分類
    SARIFResult        # SARIF報告
)

# ❌ 禁止 - 重複定義
class MySeverity(Enum):
    HIGH = "high"  # 錯誤！使用 aiva_common.Severity
```

### 🌐 多語言協作

**Python 模組**：
```python
from ..aiva_common.schemas import FunctionTaskPayload, FindingPayload
```

**Go 模組**：
```go
import "services/features/common/go/aiva_common_go"
```

**Rust 模組**：
```rust
use aiva_common::schemas::{FunctionTaskPayload, FindingPayload};
```

### 🔧 開發工具推薦

#### 多語言開發必備插件

| 語言 | 必備插件 | 開發用途 |
|------|---------|---------|
| 🐍 **Python** | Pylance + Ruff + Black | 型別檢查、快速 linting、格式化 |
| 🐹 **Go** | golang.go | gopls、除錯、測試、格式化 |
| 🦀 **Rust** | rust-analyzer | 語言伺服器、Cargo 整合、除錯 |

#### 跨語言開發推薦工具

| 功能需求 | 推薦插件 | 說明 |
|---------|---------|------|
| 🛡️ **安全掃描** | SonarLint | 支援 Python/Go，靜態安全分析 |
| 🤖 **AI 程式碼助手** | GitHub Copilot | 多語言程式碼生成與解釋 |
| 🔍 **程式碼品質** | ErrorLens + Code Spell Checker | 即時錯誤提示、拼寫檢查 |
| 🐳 **容器開發** | Docker + Dev Containers | Rust/Go 編譯環境容器化 |

#### 語言特定快速技巧

**Python (87.0% 組件)**：
- 使用 Ruff 進行超快速 linting（比 pylint 快 10-100 倍）
- Black 自動格式化：`Ctrl+Shift+I`
- 遵循 PEP 8 編碼規範

**Go (13.0% 組件)**：
- gopls 提供完整的語言支援
- 使用 `Go: Test Package` 執行測試
- 格式化自動使用 gofmt

**更多開發工具資訊**：
- 🛠️ [開發指南總覽](../../guides/development/README.md)
- 🔌 [插件與工具指南](../../guides/development/PLUGINS_AND_TOOLS_INVENTORY.md)
- 🔧 [工具集使用手冊](../../tools/README.md)

### 📝 開發流程

1. **參考現有實現**：查看類似功能模組
2. **繼承基類**：使用 `FeatureBase`（如果適用）
3. **使用標準枚舉**：從 `aiva_common` 導入
4. **提供同步接口**：`scan(target, options) -> dict`
5. **編寫測試**：參考 `testing/features/`

### ⚙️ 執行前的準備工作

**核心原則**：充分利用現有資源，避免重複造輪子

在開始任何修改或新增安全檢測功能前，務必執行以下檢查：

#### 1. 檢查本機現有工具
```bash
# 查看基礎組件
ls services/features/base/        # 功能基類、HTTP客戶端
ls services/features/common/      # 智能檢測管理器
ls testing/features/              # 測試腳本
```

#### 2. 利用 VS Code 擴展功能
```python
# Pylance MCP 工具 (推薦):
# - pylanceFileSyntaxErrors: 檢查語法錯誤
# - pylanceRunCodeSnippet: 測試 Payload 生成邏輯
# - pylanceInvokeRefactoring: 移除未使用的導入

# SonarQube 工具 (安全檢測必備):
# - sonarqube_analyze_file: 檢查代碼安全問題
# - sonarqube_list_potential_security_issues: 列出潛在漏洞
```

#### 3. 參考現有功能實現
完善的實現案例：
- `function_sqli/` - SQL 注入檢測（多引擎、智能檢測）
- `function_xss/` - XSS 檢測（DOM/Stored/Reflected）
- `function_idor/` - IDOR 檢測（垂直/水平越權）

#### 4. 功能不確定時的查詢資源
- 🌐 **安全規範**：OWASP Top 10, CWE, CAPEC
- 📚 **工具文檔**：Burp Suite, ZAP, SQLMap
- 🔍 **PoC 參考**：使用 `github_repo` 搜索公開漏洞 PoC
- 🛡️ **CVE 數據**：使用 `fetch_webpage` 查詢 CVE 詳情
- 📖 **編碼技巧**：WAF 繞過技術和 Payload 混淆方法

#### 5. 選擇最佳方案的判斷標準
- ✅ 優先繼承 `FeatureBase` 基類，複用通用邏輯
- ✅ 優先參考 OWASP 和業界公認的檢測方法
- ✅ Payload 設計參考知名安全工具（SQLMap, XSStrike 等）
- ⚠️ 避免自創檢測邏輯，容易產生誤報
- ⚠️ 新漏洞檢測方法不確定時，先查詢 CVE 和安全公告

### 🚫 保留未使用函數原則

若發現定義但未使用的函數，只要不影響運作，建議保留：
- 預留的 API 端點
- 未來功能的基礎架構
- 測試/除錯輔助函數
- 向下兼容性考量

---

## 📋 快速參考

### 語言分佈

```
🐍 Python │████████████████████████████████████████████ 87% (12,002行)
🐹 Go     │██████ 13% (1,796行)
🦀 Rust   │▌ <1% (計劃中)
```

### 技術棧

- **Python**: 主要語言（70%），asyncio 僅在 AI Commander
- **Go**: 高並發（15%），認證掃描，Cobra CLI
- **Rust**: 高性能（10%），密碼檢測，PyO3 綁定
- **TypeScript**: 動態掃描（5%）

### 整體統計

- **總代碼行數**: 13,798 行
- **檔案數量**: 87 個（75 Python + 11 Go + 1 Rust）
- **功能模組**: 17 個（5 高完成 + 4 中完成 + 2 低完成 + 5 需人工 + 1 支援）

### 外部工具

- **已集成**: sqlmap, XSStrike, NoSQLMap
- **待安裝**: nuclei, Dalfox, jwt_tool

---

## 🎨 架構圖表資源

- 📊 [功能分層架構圖](../docs/reports/architecture_diagrams/functional/FEATURES_INTEGRATED_FUNCTIONAL.mmd)
- 🛡️ [安全功能架構圖](../docs/reports/architecture_diagrams/functional/FEATURES_SECURITY_FUNCTIONS.mmd)
- 🔴 [核心功能架構圖](../docs/reports/architecture_diagrams/functional/FEATURES_CORE_FUNCTIONS.mmd)
- 📈 [多語言協作架構圖](../docs/reports/architecture_diagrams/FEATURES_MODULE_INTEGRATED_ARCHITECTURE.mmd)

---

## ⚠️ 重要開發注意事項

### 架構靈活性原則

Features 模組由眾多獨立安全檢測功能組成，**每個子功能可採用最適合其特性的內部架構**：

**✅ 必須遵守（模組層級）**：
- 使用 `aiva_common` 標準（Severity, Confidence, SARIF）
- 統一的跨模組通信接口（AivaMessage）
- 符合所用程式語言的官方規範

**🎨 完全自由（子功能內部）**：
- 內部目錄結構（扁平/分層/模塊化 皆可）
- 算法實現方式（OOP/函數式/過程式）
- 數據流設計（同步/異步/事件驅動）
- 性能優化策略（緩存/並發/流式處理）

**實際案例**：
```python
# ✅ 簡單功能 - 單文件實現
services/features/xss_detector/
  ├── detector.py          # 單文件實現
  └── patterns.json

# ✅ 複雜功能 - 分層架構  
services/features/advanced_sqli/
  ├── core/
  │   ├── engine.py
  │   └── parser.py
  ├── detectors/
  ├── utils/
  └── main.py

# ✅ 多語言混合
services/features/crypto_analyzer/
  ├── python_wrapper/      # Python 接口層
  ├── rust_engine/         # Rust 核心引擎
  └── shared_schemas/      # 共享數據定義
```

### 架構選擇指南

| 功能複雜度 | 推薦架構 | 範例 |
|-----------|---------|------|
| **簡單** (< 500 行) | 單文件/單類 | XSS 檢測, 敏感信息洩露 |
| **中等** (500-2000 行) | 模塊化分層 | SQL 注入, XXE, SSRF |
| **複雜** (> 2000 行) | 分層 + 插件 | SAST 引擎, 混合掃描器 |
| **高性能需求** | Rust/Go 核心 + Python 包裝 | 密碼學分析, 大規模爬蟲 |
| **即時處理** | 事件驅動/流式 | WebSocket 掃描, 即時監控 |

**關鍵原則**：
- 🎯 **對外統一**：必須提供標準 `execute()` 或 `scan()` 接口
- 🔓 **對內自由**：內部實現完全由開發者決定
- 📊 **結果標準**：輸出必須符合 SARIF 2.1.0 + aiva_common 枚舉
- 🌐 **語言規範**：遵循所用語言的官方最佳實踐

---

## 📚 相關開發指南

### 核心開發資源

- 🏗️ **[AIVA Common 共享庫](../aiva_common/README.md)** - 統一數據模型、枚舉定義、命令系統
- 🛠️ **[開發指南總覽](../../guides/development/README.md)** - 完整開發指南索引
- 🏗️ **[架構指南](../../guides/architecture/README.md)** - v2.0 數據合約架構文檔
- 🔧 **[工具集使用手冊](../../tools/README.md)** - 專業工具操作
- 📊 **[Services 架構總覽](../README.md)** - 六大核心服務架構

### 推薦閱讀順序

**新手開發者**：
1. [開發指南總覽](../../guides/development/README.md) - 了解整體開發流程
2. [AIVA Common](../aiva_common/README.md) - 學習數據模型使用
3. [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md) - Features 模組架構設計

**進階開發者**：
1. [架構指南](../../guides/architecture/README.md) - 深入架構設計
2. [Services 總覽](../README.md) - 掌握服務架構
3. [工具集手冊](../../tools/README.md) - 提升開發效率

---

**文件版本**: v6.2  
**最後更新**: 2025-12-12  
**維護團隊**: AIVA Multi-Language Architecture Team

*這是 AIVA Features 模組的主要導航文件。查看 [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md) 了解詳細架構設計。*
