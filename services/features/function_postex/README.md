# function_postex - 後滲透測試模組

> **版本**: v3.0.0 | **狀態**: 🔧 開發中 (70%) | **語言**: Python

## 🎯 模組概述

後滲透階段（Post-Exploitation）的安全檢測與分析模組，專注於已獲得初始存取後的權限提升、橫向移動與持久化路徑偵測。

> ⚠️ **警告**: 僅限授權滲透測試使用。本模組預設以安全模式執行（僅偵測設定與路徑，不執行實際的越權或竄改）。

### 功能清單

| 功能 | 說明 |
|------|------|
| 權限提升檢測 (PrivEsc) | Linux: SUID/Sudo/Kernel/Docker；Windows: 服務路徑/AIE/Token |
| 橫向移動分析 (Lateral) | 內網主機發現、SMB/SSH/RDP/WinRM 服務列舉 |
| 持久化偵測 (Persistence) | Linux: Cron/Systemd/SSH/LD_PRELOAD；Windows: 登錄檔/排程/服務 |

## 📐 架構設計

```
function_postex/
├── __init__.py                      # 模組入口匯出
├── detector/
│   └── postex_detector.py           # 整合主入口 (PostExDetector)
├── detectors/                       # (相容層，統一指向 detector)
│   ├── __init__.py
│   └── postex_detector.py
└── engines/
    ├── __init__.py
    ├── privilege_escalation.py      # 結構化 PrivEscEngine
    ├── lateral_movement.py          # 結構化 LateralEngine
    ├── persistence.py               # 結構化 PersistenceEngine
    ├── privilege_engine.py          # 整合型 PrivilegeEscalator
    ├── lateral_engine.py            # 整合型 LateralMovementTester
    └── persistence_engine.py        # 整合型 PersistenceChecker
```

## 🚀 執行方式

### 透過整合型 Detector 呼叫

```python
from services.features.function_postex import PostExDetector

detector = PostExDetector()

# 權限提升檢測（安全模式）
findings = detector.analyze(
    test_type="privilege_escalation",
    target="localhost",
    task_id="task-001",
    scan_id="scan-001",
    safe_mode=True,
)
```

### 直接使用結構化引擎

```python
from services.features.function_postex.engines import PrivilegeEscalationEngine

# 權限提升分析
priv_engine = PrivilegeEscalationEngine(safe_mode=True)
vectors = priv_engine.scan(target="localhost")
```

## 🔧 內部實作細節

### 1. 權限提升檢測
- **Linux**: 分析 `find/vim/nmap` 等危險 SUID、檢查 Sudo `NOPASSWD` 誤配置、檢查可寫關鍵檔案 (`/etc/passwd`)、Docker Socket。
- **Windows**: 檢查 Unquoted Service Paths、AlwaysInstallElevated 機碼、危險 Token 權限等。

### 2. 持久化偵測
- **Linux**: 檢查 `/etc/crontab`、`~/.bashrc`、`~/.ssh/` 是否可寫入。
- **Windows**: 檢查 Registry Run Keys、Startup 資料夾。

## 🔒 授權與安全模式
- 預設 `safe_mode=True`，只透過靜態配置檔案分析，不建立檔案或修改設定。
- 若 `safe_mode=False`，則允許執行可能改變系統狀態的提權 PoC 指令 (必須明確傳遞授權 Token)。
- 未獲授權的系統測試屬違法行為。
