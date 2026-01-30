# 外部模組執行模式分析報告

**生成日期**: 2026-01-13  
**檢查範圍**: services/features/function_* 模組  
**架構說明**: 3 種執行模式（非必需全部實現）

---

## 🎯 重要澄清

### 3 種執行模式（互相獨立，非必需）

#### 1. Direct Import 模式 ⭐ (這就是 CLI 實現！)
```python
from services.features.function_xss.detector import XSSDetector
detector = XSSDetector()
result = detector.scan(url)
```
- ✅ **這就是 CLI 接口** - 直接導入使用
- ✅ **最簡單直接** - 無需額外架構
- ✅ **適用所有場景** - 腳本、測試、整合

#### 2. CommandHandler 模式（可選）
```python
from services.features.function_xss import XSSCommandHandler
handler = XSSCommandHandler()
result = await handler.handle_command(command)
```
- ⚠️ **非必需** - 只為 AI 統一調度提供便利
- ⚠️ **額外開銷** - 需要實現額外的包裝層
- 用途：AI Command Center 統一調度

#### 3. Worker 模式（可選）
- ⚠️ **非必需** - 只為背景異步任務
- 用途：監聽 RabbitMQ，背景處理

### 結論
- ❌ **不是所有模組都需要 CommandHandler**
- ✅ **有 Detector 類就是完整的 CLI 實現**
- ✅ **CommandHandler 只是可選的包裝層**

---

## 📊 總體狀態（基於 Direct Import）

| 狀態 | 模組數 | 百分比 | 說明 |
|------|--------|--------|------|
| ✅ **Direct Import Ready** | ? | ?% | 有 Detector 類可直接使用 |
| 🎁 **額外有 CommandHandler** | 5 | 31.25% | 同時支援 AI 調度 |
| ⚠️ **需要檢查** | 11 | 68.75% | 需確認 Detector 類是否存在 |
| **總計** | **16** | **100%** | 所有功能模組 |

---

## ✅ 已完成模組 (5/16)

### 1. function_xss - XSS 漏洞檢測
**狀態**: ✅ 完整實現

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ✅ | `XSSCommandHandler` |
| handle_command 方法 | ✅ | 完整實現 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_XSS_TEST` |
| 檔案位置 | ✅ | `services/features/function_xss/command_handler.py` |

**支援的命令**:
```python
CommandType.FEATURE_XSS_TEST = "feature_xss_test"
```

**使用範例**:
```python
from services.features.function_xss import XSSCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

handler = XSSCommandHandler()
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={
        "target_url": "https://example.com",
        "test_type": "reflected",  # reflected/stored/dom
        "parameters": ["q", "search"]
    }
)
result = await handler.handle_command(command)
```

---

### 2. function_sqli - SQL 注入檢測
**狀態**: ✅ 完整實現

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ✅ | `SQLiCommandHandler` |
| handle_command 方法 | ✅ | 完整實現 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_SQLI_TEST` |
| 檔案位置 | ✅ | `services/features/function_sqli/command_handler.py` |

**支援的命令**:
```python
CommandType.FEATURE_SQLI_TEST = "feature_sqli_test"
```

**使用範例**:
```python
from services.features.function_sqli import SQLiCommandHandler

handler = SQLiCommandHandler()
command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    payload={
        "target_url": "https://example.com/api",
        "injection_points": ["id", "user"],
        "db_type": "mysql"  # mysql/postgresql/mssql/oracle
    }
)
result = await handler.handle_command(command)
```

---

### 3. function_ssrf - SSRF 漏洞檢測
**狀態**: ✅ 完整實現

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ✅ | `SSRFCommandHandler` |
| handle_command 方法 | ✅ | 完整實現 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_SSRF_TEST` |
| 檔案位置 | ✅ | `services/features/function_ssrf/command_handler.py` |

**支援的命令**:
```python
CommandType.FEATURE_SSRF_TEST = "feature_ssrf_test"
```

**特色**:
- ✅ 支援同步/異步雙重執行路徑
- ✅ 內網 IP 探測
- ✅ 雲端元數據服務檢測 (AWS/Azure/GCP)
- ✅ 協議走私檢測

---

### 4. function_idor - IDOR 漏洞檢測
**狀態**: ✅ 完整實現

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ✅ | `IDORCommandHandler` |
| handle_command 方法 | ✅ | 完整實現 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_IDOR_TEST` |
| 檔案位置 | ✅ | `services/features/function_idor/command_handler.py` |

**支援的命令**:
```python
CommandType.FEATURE_IDOR_TEST = "feature_idor_test"
```

---

### 5. function_bizlogic - 業務邏輯漏洞
**狀態**: ✅ 完整實現

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ✅ | `BizLogicCommandHandler` |
| handle_command 方法 | ✅ | 完整實現 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_BIZLOGIC_TEST` |
| 檔案位置 | ✅ | `services/features/function_bizlogic/command_handler.py` |

**支援的命令**:
```python
CommandType.FEATURE_BIZLOGIC_TEST = "feature_bizlogic_test"
```

---

## ⚠️ 待實現模組 (11/16)

### Python 模組 (需要 CommandHandler)

#### 1. function_wordlist_generator - 字典生成器
**狀態**: ⚠️ 部分完成

| 項目 | 狀態 | 詳情 |
|------|------|------|
| CommandHandler 類 | ⚠️ | 有 `WordlistGeneratorCommandHandler` 但可能未完整 |
| CommandType 定義 | ✅ | `CommandType.FEATURE_WORDLIST_GENERATE` |
| 建議 | - | 檢查 handle_command 實現是否完整 |

#### 2. function_info_leak - 信息洩漏檢測
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `InfoLeakCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 高 | 信息洩漏是常見高危漏洞 |

**建議的 CommandType**:
```python
FEATURE_INFO_LEAK_TEST = "feature_info_leak_test"
```

#### 3. function_postex - 後滲透
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `PostExCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 中 | 滲透測試後期階段使用 |

**建議的 CommandType**:
```python
FEATURE_POSTEX = "feature_postex"
```

#### 4. function_forensic - 取證分析
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `ForensicCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 低 | 專業取證場景 |

**建議的 CommandType**:
```python
FEATURE_FORENSIC_ANALYSIS = "feature_forensic_analysis"
```

#### 5. function_reverse_engineering - 逆向工程
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `ReverseEngineeringCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 低 | 專業逆向場景 |

**建議的 CommandType**:
```python
FEATURE_REVERSE_ENGINEERING = "feature_reverse_engineering"
```

#### 6. function_social_engineering - 社會工程
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `SocialEngineeringCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 中 | 釣魚測試常用 |

**建議的 CommandType**:
```python
FEATURE_SOCIAL_ENGINEERING = "feature_social_engineering"
```

#### 7. function_steganography - 隱寫術
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `SteganographyCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 低 | 專業場景 |

**建議的 CommandType**:
```python
FEATURE_STEGANOGRAPHY = "feature_steganography"
```

#### 8. function_web_scanner - Web 掃描器
**狀態**: ❌ 未實現

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| CommandHandler | ❌ | 需創建 `WebScannerCommandHandler` |
| CommandType | ❌ | 需添加到 `commands.py` |
| 優先級 | 高 | 基礎掃描功能 |
| 備註 | - | 可能與 scan 模組重複？需確認 |

**建議的 CommandType**:
```python
FEATURE_WEB_SCAN = "feature_web_scan"
```

---

### 非 Python 模組 (使用 Direct Import 模式)

#### 9. function_authn_go - Go 身份驗證
**狀態**: ✅ 正確架構 (不需要 Python CommandHandler)

| 項目 | 狀態 | 說明 |
|------|------|------|
| 架構模式 | ✅ | Direct Import (Go 模組) |
| 使用方式 | ✅ | `from function_authn_go import AuthnDetector` |
| CommandHandler | N/A | Go 模組不需要 Python CommandHandler |
| 流程數 | ✅ | 4 個數據流 |

**正確使用方式**:
```python
# Direct Import - 直接導入使用
from services.features.function_authn_go import AuthnDetector

detector = AuthnDetector()
result = detector.analyze(target_url)
```

#### 10. function_crypto - Rust 加密分析
**狀態**: ✅ 正確架構 (不需要 Python CommandHandler)

| 項目 | 狀態 | 說明 |
|------|------|------|
| 架構模式 | ✅ | Direct Import (Rust 模組) |
| 使用方式 | ✅ | CLI 直接調用 |
| CommandHandler | N/A | Rust 模組不需要 Python CommandHandler |
| 流程數 | ✅ | 4 個數據流 |

**正確使用方式**:
```bash
# CLI 直接調用
cd services/features/function_crypto
cargo run -- analyze-cookies --cookies-json cookies.json
cargo run -- analyze-headers --headers-json headers.json
cargo run -- scan-javascript --js-file app.js
cargo run -- analyze-tls --url https://example.com
```

#### 11. function_exploit - 漏洞利用
**狀態**: ⚠️ 無 __init__.py (模組不完整)

| 項目 | 狀態 | 建議動作 |
|------|------|---------|
| __init__.py | ❌ | 需創建基礎檔案 |
| 模組結構 | ❌ | 需確定是 Python/Rust/Go |
| 優先級 | 高 | 核心攻擊功能 |

---

## 📋 CommandType 完整清單

### 已在 commands.py 定義的 CommandType

```python
# Scan 模組 (由 scan 引擎處理)
SCAN_PHASE0 = "scan_phase0"
SCAN_PHASE1 = "scan_phase1"
SCAN_COMPREHENSIVE = "scan_comprehensive"

# Feature 模組 (已實現 ✅)
FEATURE_XSS_TEST = "feature_xss_test"                   # ✅ XSS
FEATURE_SQLI_TEST = "feature_sqli_test"                 # ✅ SQLi
FEATURE_SSRF_TEST = "feature_ssrf_test"                 # ✅ SSRF
FEATURE_IDOR_TEST = "feature_idor_test"                 # ✅ IDOR
FEATURE_BIZLOGIC_TEST = "feature_bizlogic_test"         # ✅ 業務邏輯

# Feature 模組 (已定義但可能未完全實現 ⚠️)
FEATURE_PAYLOAD_GENERATE = "feature_payload_generate"   # ⚠️ Payload 生成器
FEATURE_WORDLIST_GENERATE = "feature_wordlist_generate" # ⚠️ 字典生成器
```

### 建議新增的 CommandType

```python
# 高優先級 (常用功能)
FEATURE_INFO_LEAK_TEST = "feature_info_leak_test"       # 信息洩漏檢測
FEATURE_WEB_SCAN = "feature_web_scan"                   # Web 掃描

# 中優先級
FEATURE_SOCIAL_ENGINEERING = "feature_social_engineering"  # 社會工程
FEATURE_POSTEX = "feature_postex"                          # 後滲透

# 低優先級 (專業場景)
FEATURE_FORENSIC_ANALYSIS = "feature_forensic_analysis"    # 取證分析
FEATURE_REVERSE_ENGINEERING = "feature_reverse_engineering" # 逆向工程
FEATURE_STEGANOGRAPHY = "feature_steganography"            # 隱寫術
FEATURE_EXPLOIT = "feature_exploit"                        # 漏洞利用
```

---

## 🎯 實現優先級建議

### Phase 1: 核心功能補全 (高優先級)
1. ✅ **function_info_leak** - 信息洩漏是最常見的漏洞
2. ✅ **function_web_scanner** - 基礎掃描功能（如不與 scan 重複）
3. ⚠️ **function_wordlist_generator** - 檢查並完善現有實現

### Phase 2: 擴展功能 (中優先級)
4. **function_social_engineering** - 釣魚測試常用
5. **function_postex** - 滲透測試後期重要
6. **function_exploit** - 確定架構並實現

### Phase 3: 專業功能 (低優先級)
7. **function_forensic** - 專業取證場景
8. **function_reverse_engineering** - 專業逆向場景
9. **function_steganography** - 特定場景使用

---

## 📐 CommandHandler 實現模板

### 標準實現模板

```python
"""
模組名稱 CommandHandler
實現 aiva_common.CommandHandler 協議
"""

from typing import Optional
from services.aiva_common.command_center import CommandHandler
from services.aiva_common.schemas import AICommand, AICommandResult, CommandContext, CommandStatus
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class YourModuleCommandHandler(CommandHandler):
    """
    [模組名稱] 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 FEATURE_XXX_TEST 命令。
    """
    
    def __init__(self):
        """初始化處理器"""
        self.logger = logger
        # 初始化你的檢測器
        # self.detector = YourDetector()
    
    async def handle_command(
        self, 
        command: AICommand, 
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理命令
        
        Args:
            command: AI 命令
            context: 執行上下文（可選）
            
        Returns:
            命令執行結果
        """
        try:
            self.logger.info(f"🔍 處理命令: {command.command_type}")
            
            # 1. 驗證 payload
            payload = command.payload or {}
            if not payload.get("target_url"):
                return AICommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.FAILED,
                    error="缺少必要參數: target_url"
                )
            
            # 2. 執行檢測邏輯
            # result = await self.detector.scan(payload["target_url"])
            
            # 3. 返回結果
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.SUCCESS,
                data={
                    "findings": [],  # 你的檢測結果
                    "summary": "檢測完成"
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ 命令執行失敗: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                error=str(e)
            )
```

### 添加到 commands.py

```python
# 在 services/aiva_common/schemas/commands.py 添加

class CommandType(str, Enum):
    # ... 現有定義 ...
    
    # 新增你的 CommandType
    FEATURE_YOUR_MODULE_TEST = "feature_your_module_test"  # 你的模組說明
```

---

## 📊 進度追蹤

### 實現進度

- ✅ **已完成**: 5/16 (31.25%)
- ⚠️ **部分完成**: 1/16 (6.25%)
- ❌ **未開始**: 8/16 (50%)
- ✅ **正確架構** (非 Python): 2/16 (12.5%)

### 目標

- **短期目標** (1-2 週): 完成 Phase 1 核心功能 (3 個模組)
- **中期目標** (1 個月): 完成 Phase 2 擴展功能 (3 個模組)
- **長期目標** (2 個月): 完成所有 Python 模組的 CommandHandler

---

## 📝 注意事項

### 架構原則

1. **Python 功能模組** → 需要 CommandHandler
   - 實現 `handle_command()` 方法
   - 在 `commands.py` 添加 CommandType
   - 在 `__init__.py` 導出 Handler

2. **Rust/Go/TypeScript 模組** → Direct Import 模式
   - 不需要 Python CommandHandler
   - CLI 直接調用或 Python wrapper
   - 保持語言原生性能優勢

3. **統一接口**
   - 所有 CommandHandler 實現相同協議
   - 統一的錯誤處理和日誌
   - 統一的返回格式 (AICommandResult)

### 質量要求

- ✅ 完整的類型標註
- ✅ 詳細的文檔字串
- ✅ 錯誤處理和日誌
- ✅ 參數驗證
- ✅ 單元測試

---

**報告結束**
