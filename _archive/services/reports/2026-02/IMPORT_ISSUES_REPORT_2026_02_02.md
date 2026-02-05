# AIVA Services 導入問題報告

**生成日期:** 2026-02-02  
**掃描範圍:** `C:\D\fold7\AIVA-git\services`

---

## 📋 報告摘要

| 類別 | 問題數量 |
|------|----------|
| 缺失的 Python 檔案 | 20 |
| 缺失的目錄/模組 | 5 |
| __init__.py 導入錯誤 | 10 個檔案 |
| 相對導入路徑錯誤 | 3 |

---

## 🔴 1. 缺失的檔案（導入語句指向不存在的檔案）

### 1.1 features/function_* 模組缺失的 command_handler.py

| 模組路徑 | 缺失檔案 | 導入位置 |
|----------|----------|----------|
| `features/function_sqli/` | `command_handler.py` | `__init__.py:19` |
| `features/function_bizlogic/` | `command_handler.py` | `__init__.py:29` |

**影響:** 這些模組的 CommandHandler 無法被導入，CLI 整合功能將失敗。

**詳細說明:**
```
features\function_sqli\__init__.py:19
  from .command_handler import SQLiCommandHandler
  -> 預期檔案: features\function_sqli\command_handler.py (不存在)

features\function_bizlogic\__init__.py:29
  from .command_handler import BizLogicCommandHandler
  -> 預期檔案: features\function_bizlogic\command_handler.py (不存在)
```

**備註:** `function_xss/command_handler.py` 是唯一存在的 command_handler，可作為參考模板。

---

### 1.2 aiva_common/ai/ 缺失的核心 AI 組件

| 缺失檔案 | 導入位置 | 預期導出的類別 |
|----------|----------|----------------|
| `dialog_assistant.py` | `__init__.py:63, 284, 317` | `AIVADialogAssistant`, `DialogIntent` |
| `plan_executor.py` | `__init__.py:69, 319` | `AIVAPlanExecutor`, `ExecutionConfig` |
| `experience_manager.py` | `__init__.py:75, 273, 318` | `AIVAExperienceManager`, `LearningSession`, `create_experience_manager` |
| `capability_evaluator.py` | `__init__.py:85, 262, 315` | `AIVACapabilityEvaluator`, `CapabilityEvidence`, `create_capability_evaluator` |
| `cross_language_bridge.py` | `__init__.py:95, 316` | `AIVACrossLanguageBridge`, `BridgeConfig` |
| `rag_agent.py` | `__init__.py:101, 320` | `BioNeuronRAGAgent`, `RAGConfig` |
| `skill_graph_analyzer.py` | `__init__.py:107, 321` | `AIVASkillGraphAnalyzer`, `SkillNode` |
| `integration_manager.py` | `__init__.py:113` | `AIIntegrationManager`, `AITask`, `AIResult`, ... |

**影響:** 這些是 AI 共享組件的核心實現，缺失將導致 AI 功能降級。

**現有檔案:** `interfaces.py`, `performance_config.py`, `registry.py`, `__init__.py`

**處理方式:** `__init__.py` 使用 `contextlib.suppress(ImportError)` 包裝，運行時不會崩潰，但功能不可用。

---

### 1.3 core/aiva_core/service_backbone/ 缺失的服務檔案

| 目錄 | 缺失檔案 | 導入位置 | 預期導出 |
|------|----------|----------|----------|
| `system/` | `resource_watchdog.py` | `__init__.py:3` | `ResourceWatchdog`, `ResourceStatus`, `ResourceThresholds` |
| `utils/` | `config.py` | `__init__.py:7` | `AIVAConfig`, `ConfigManager` |
| `utils/` | `ai_identifier.py` | `__init__.py:8` | `AIIdentifier`, `AISignature` |

**現有 utils/ 檔案:** `logging_formatter.py`, `repair_tool.py`, `__init__.py`

---

### 1.4 core/aiva_core/task_planning/persistence/ 缺失的持久化檔案

| 缺失檔案 | 導入位置 | 預期導出 |
|----------|----------|----------|
| `task_storage.py` | `__init__.py:12` | `TaskStorage`, `TaskStatus` |
| `task_manager.py` | `__init__.py:13` | `TaskManager` |

**現有檔案:** 僅 `__init__.py`

**影響:** 任務持久化和斷點續傳功能 (P0-3) 無法使用。

---

### 1.5 aiva_common/schemas/testing/ 缺失的測試 Schema

| 缺失檔案 | 導入位置 | 預期導出 |
|----------|----------|----------|
| `api_testing.py` | `__init__.py:13` | API 測試相關的 schemas |

**現有檔案:** `scenarios.py`, `tasks.py`, `__init__.py`

---

### 1.6 core/aiva_core/service_backbone/coordination/ 缺失的插件

| 缺失路徑 | 導入位置 | 預期內容 |
|----------|----------|----------|
| `plugins/ai_summary_plugin.py` | `ai_controller.py:16` | `AISummaryPlugin` 類別 |

**處理方式:** 代碼使用 try/except 處理，但插件功能不可用。

---

### 1.7 features/function_exploit/executor/ 缺失的遺留模組

| 缺失檔案 | 導入位置 | 預期導出 |
|----------|----------|----------|
| `exploit_manager_legacy.py` | `attack_executor.py:397` | `ExploitManager` |

**現有檔案:** `attack_executor.py`, `bizlogic_attack_executor.py`

---

## 🟡 2. 相對導入路徑錯誤

### 2.1 scenarios.py 導入路徑問題

**檔案:** `aiva_common/schemas/testing/scenarios.py:10`
```python
from ...enhanced import (
    EnhancedScanRequest,
    EnhancedTaskExecution,
)
```

**問題:** 導入路徑 `...enhanced` 指向 `aiva_common/enhanced.py`，但該檔案不存在。

**實際位置:** `aiva_common/schemas/enhanced.py` 存在

**修正建議:** 
```python
from ..enhanced import (
    EnhancedScanRequest,
    EnhancedTaskExecution,
)
```

---

### 2.2 features_ready/function_sqli 導入問題 ✅ 已解決

**檔案:** `features/features_ready/function_sqli/command_handler.py:43` (已刪除)
**解決方案:** 2026-02-03 移除整個 features_ready 目錄（僅包含 2 個過時檔案）
```python
from .integration_tools.sql_tools import SQLInjectionManager, SQLTarget
```

**問題:** `integration_tools/` 目錄可能不存在或結構不完整。

---

## 🟠 3. 缺失的目錄/模組

| 預期模組路徑 | 狀態 | 說明 |
|--------------|------|------|
| `features/function_ddos/` | ❌ 不存在 | DDoS 測試模組 |
| `features/function_exploit_framework/` | ❌ 不存在 | 漏洞利用框架 |
| `features/function_payload_generator/` | ❌ 不存在 | 載荷生成器 |
| `features/function_exploit/__init__.py` | ❌ 不存在 | 有目錄但無 init |
| `features/function_infoleak/__init__.py` | ❌ 不存在 | 空目錄 |
| `core/aiva_core/service_backbone/coordination/plugins/` | ❌ 不存在 | 插件目錄 |

---

## 🔵 4. function_* 模組 command_handler.py 狀態總覽

| 模組 | command_handler.py | 狀態 |
|------|-------------------|------|
| function_xss | ✅ 存在 | 正常 |
| function_sqli | ❌ 不存在 | **需要創建** |
| function_bizlogic | ❌ 不存在 | **需要創建** |
| function_ssrf | ❌ 不存在 | 待開發 |
| function_forensic | ❌ 不存在 | 待開發 |
| function_crypto | ❌ 不存在 | Rust CLI 架構，不需要 |
| function_postex | ❌ 不存在 | 待開發 |
| function_wordlist_generator | ❌ 不存在 | 有 handler.py |
| function_idor | ❌ 不存在 | 待開發 |
| function_steganography | ❌ 不存在 | 待開發 |
| function_reverse_engineering | ❌ 不存在 | 待開發 |
| function_web_scanner | ❌ 不存在 | 待開發 |
| function_social_engineering | ❌ 不存在 | 待開發 |
| function_info_leak | ❌ 不存在 | 待開發 |
| function_authn_go | ❌ 不存在 | Go CLI 架構 |
| function_exploit | ❌ 不存在 | 無 __init__.py |

---

## 📊 5. 問題嚴重度分類

### 🔴 高優先級（會導致 ImportError）

1. **features/function_sqli/__init__.py** - 嘗試導入不存在的 `command_handler.py`
2. **features/function_bizlogic/__init__.py** - 嘗試導入不存在的 `command_handler.py`
3. **core/aiva_core/task_planning/persistence/__init__.py** - 缺失 `task_storage.py`, `task_manager.py`
4. **core/aiva_core/service_backbone/system/__init__.py** - 缺失 `resource_watchdog.py`
5. **core/aiva_core/service_backbone/utils/__init__.py** - 缺失 `config.py`, `ai_identifier.py`
6. **aiva_common/schemas/testing/__init__.py** - 缺失 `api_testing.py`

### 🟡 中優先級（使用 suppress/try-except 處理）

1. **aiva_common/ai/__init__.py** - 8 個 AI 組件檔案缺失（已用 contextlib.suppress 包裝）
2. **core/aiva_core/service_backbone/coordination/ai_controller.py** - 插件導入失敗

### 🟢 低優先級（功能預留或架構變更）

1. **features/function_exploit/executor/attack_executor.py** - 遺留模組導入
2. **aiva_common/schemas/testing/scenarios.py** - 相對導入路徑問題

---

## 📝 6. 建議修復順序

1. **立即修復 (P0):**
   - 為 `function_sqli/__init__.py` 和 `function_bizlogic/__init__.py` 添加 try/except 或創建 command_handler.py
   - 創建 `task_planning/persistence/task_storage.py` 和 `task_manager.py` 骨架

2. **短期修復 (P1):**
   - 創建 `service_backbone/system/resource_watchdog.py`
   - 創建 `service_backbone/utils/config.py` 和 `ai_identifier.py`
   - 創建 `schemas/testing/api_testing.py`

3. **中期修復 (P2):**
   - 實現 `aiva_common/ai/` 中的 AI 組件
   - 修正 `scenarios.py` 的導入路徑

4. **長期規劃 (P3):**
   - 完善所有 function_* 模組的 command_handler.py
   - 創建缺失的功能模組目錄

---

## 🔧 7. 快速修復腳本

```python
# 用於添加安全導入包裝的腳本
import_wrapper = '''
try:
    from .command_handler import {handler_class}
except ImportError:
    {handler_class} = None
'''

# 用於創建空骨架檔案的腳本
skeleton_content = '''"""
{module_name} - 骨架檔案

TODO: 實現此模組的功能
"""

__all__ = []
'''
```

---

*報告生成完畢。建議優先處理高優先級問題以確保系統穩定性。*
