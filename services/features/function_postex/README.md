# function_postex - 後滲透測試模組

> **版本**: v2.0.0 | **狀態**: 🔧 開發中 (70%) | **語言**: Python | **更新**: 2026-03-17

## 🎯 模組概述

後滲透階段（Post-Exploitation）的安全檢測與分析模組，專注於已獲得初始存取後的權限提升、橫向移動與持久化路徑偵測。

> ⚠️ **警告**: 僅限授權滲透測試使用。本模組預設以安全模式執行（僅偵測，不執行）。

### 功能完成狀態

| 功能 | Linux | Windows | 說明 |
|------|-------|---------|------|
| 權限提升檢測 | ✅ 完成 | ✅ 完成 | Linux: SUID/Sudo/Kernel/Docker；Windows: 服務路徑/AIE/Token |
| 橫向移動分析 | ✅ 完成 | ✅ 完成 | 主機發現、服務列舉（SMB/SSH/RDP/WinRM） |
| 持久化偵測 | ✅ 完成 | ✅ 完成 | Linux: Cron/Systemd/SSH/LD_PRELOAD；Windows: 登錄檔/排程/服務 |
| 敏感資料收集 | ⏳ 待實作 | ⏳ 待實作 | 設定檔憑證、瀏覽器密碼、金鑰提取 |
| 防禦規避 | ⏳ 待實作 | ⏳ 待實作 | AV/EDR 繞過建議、日誌清除偵測 |

## 📐 實際架構

```
function_postex/
├── detector/
│   └── postex_detector.py          # 主入口（PostExDetector）
├── detectors/                       # 相容層（重新匯出 detector/）
│   ├── __init__.py
│   └── postex_detector.py
├── engines/
│   ├── __init__.py                  # 匯出兩組引擎
│   ├── privilege_escalation.py      # PrivilegeEscalationEngine（結構化）
│   ├── lateral_movement.py          # LateralMovementEngine（結構化）
│   ├── persistence.py               # PersistenceEngine（結構化）
│   ├── privilege_engine.py          # PrivilegeEscalator（整合型）
│   ├── lateral_engine.py            # LateralMovementTester（整合型）
│   └── persistence_engine.py        # PersistenceChecker（整合型）
├── __init__.py                      # 模組入口
└── README.md
```

> **架構說明**：引擎分兩組。
> - **結構化引擎**（`privilege_escalation.py` 等）：回傳 dataclass Vector，適合程式化處理。
> - **整合型引擎**（`privilege_engine.py` 等）：回傳 dict，與 AIVA FindingPayload 整合。
> - `detectors/` 為舊路徑相容層，統一指向 `detector/postex_detector.py`。

## 🚀 快速開始

### 基本使用

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

for f in findings:
    print(f"[{f.get('severity')}] {f.get('title')}")
    print(f"  {f.get('description')}")
```

### 直接使用結構化引擎

```python
from services.features.function_postex.engines import (
    PrivilegeEscalationEngine,
    LateralMovementEngine,
    PersistenceEngine,
)

# 權限提升
priv_engine = PrivilegeEscalationEngine(safe_mode=True)
vectors = priv_engine.scan(target="localhost")
for v in vectors:
    print(f"[{v.severity}] {v.title}")
    print(f"  分類: {v.category}")
    print(f"  建議: {v.recommendation}")

# 持久化路徑
persist_engine = PersistenceEngine(safe_mode=True)
vectors = persist_engine.scan()
for v in vectors:
    print(f"[{v.severity}] {v.title} ({v.technique})")
```

## 🔧 功能詳解

### 1. 權限提升檢測（PrivilegeEscalationEngine）

#### Linux 偵測項目
| 檢查 | 說明 | 對應方法 |
|------|------|---------|
| SUID/SGID 二進位 | 偵測 find/vim/nmap/perl/python 等危險 SUID | `_check_suid_binaries()` |
| Sudo 誤配置 | `NOPASSWD`、`(ALL) ALL`、vim/less/more 逃脫 | `_check_sudo_config()` |
| 可寫關鍵檔案 | `/etc/passwd`、`/etc/shadow`、`/etc/sudoers` | `_check_writable_paths()` |
| Cron Jobs | 可寫的 crontab 設定 | `_check_cron_jobs()` |
| Docker Socket | `/var/run/docker.sock` 可存取 | `_check_docker_socket()` |
| 內核版本 | 比對已知 CVE（DirtyCow、eBPF 等） | `_check_kernel_version()` |

#### Windows 偵測項目
| 檢查 | 說明 | 對應方法 |
|------|------|---------|
| Unquoted Service Paths | 含空格未加引號的服務路徑 | `_check_unquoted_service_paths()` |
| AlwaysInstallElevated | HKLM/HKCU 登錄機碼值為 0x1 | `_check_always_install_elevated()` |
| 可寫服務二進位 | 可被覆蓋的服務執行檔 | `_check_writable_service_binaries()` |
| 排程工作 | SYSTEM 身份執行且二進位可寫 | `_check_windows_scheduled_tasks()` |
| 危險 Token 權限 | SeImpersonate/SeDebug/SeBackup 等 | `_check_windows_token_privileges()` |

### 2. 橫向移動分析（LateralMovementEngine）

```python
lateral_engine = LateralMovementEngine(safe_mode=True)
vectors = lateral_engine.scan(target="192.168.1.0/24")
```

**偵測內容**：
- 內網主機發現（ICMP/ARP 掃描）
- 常見服務列舉：SMB (445)、RDP (3389)、SSH (22)、WinRM (5985)
- 弱認證服務識別

### 3. 持久化偵測（PersistenceEngine）

#### Linux 持久化機會
| 技術 | 偵測條件 |
|------|---------|
| Cron | `/etc/crontab`、`/etc/cron.d` 可寫 |
| Systemd | `/etc/systemd/system` 可寫 |
| Shell RC | `~/.bashrc`、`~/.zshrc`、`/etc/profile` 可寫 |
| SSH Keys | `~/.ssh/` 目錄可寫 |
| LD_PRELOAD | `/etc/ld.so.preload` 可寫 |

#### Windows 持久化機會
| 技術 | 偵測條件 |
|------|---------|
| Registry Run Keys | HKCU Run/RunOnce（永遠可寫）；HKLM Run（偵測現有項目） |
| Startup Folder | `%APPDATA%\...\Startup` 或 `%PROGRAMDATA%\...\Startup` 可寫 |
| Scheduled Tasks | `schtasks` 可存取（可能建立新工作） |
| Windows Services | `sc query` 可存取（可能建立新服務） |

## 📊 輸出格式

結構化引擎回傳 dataclass，主要欄位：

```python
@dataclass
class PrivEscVector:
    id: str                    # 例: "PRIVESC-SUID-FIND"
    severity: Severity         # CRITICAL / HIGH / MEDIUM / LOW
    confidence: Confidence     # HIGH / MEDIUM / LOW
    title: str
    description: str
    evidence: Dict[str, Any]   # 具體證據資料
    recommendation: str
    exploit_commands: List[str] # 示範指令（safe_mode 下僅供參考）
    category: str              # suid/sudo/kernel/cron/docker/...
```

整合型引擎（`PostExDetector.analyze()`）回傳 `List[Dict]`，格式與 AIVA `FindingPayload` 相容。

## 🔒 安全模式

| 模式 | `safe_mode=True`（預設） | `safe_mode=False` |
|------|---------|----------|
| 執行方式 | 僅偵測設定與路徑 | 可執行實際測試指令 |
| 系統變更 | 不建立任何檔案或修改設定 | 依授權範圍執行 |
| 需要授權 | 否 | 是（需 `auth_token`） |

## ⏳ 待實作功能

以下功能尚未實作，**不應**在文件中標記為完成：

- **敏感資料收集**：SSH 私鑰提取、瀏覽器密碼、設定檔憑證、Kerberos 票據
- **防禦規避**：AV/EDR 繞過建議、日誌清除偵測
- **WMI 事件訂閱**（Windows 持久化）
- **DLL 劫持偵測**（Windows 持久化）
- **Kerberoasting / Pass-the-Hash**（橫向移動進階）

## 🎯 適用場景

✅ **滲透測試** — 後滲透階段系統評估（需書面授權）
✅ **Red Team** — 橫向移動和提權路徑發現
✅ **安全審計** — 系統設定安全性稽核
✅ **合規驗證** — 特權存取控制驗證

❌ **不適用於**：
- 未獲授權的系統測試
- 生產環境全自動掃描
- Bug Bounty（大多數平台禁止後滲透測試）

## 🔗 相關標準

- [MITRE ATT&CK: Privilege Escalation (TA0004)](https://attack.mitre.org/tactics/TA0004/)
- [MITRE ATT&CK: Lateral Movement (TA0008)](https://attack.mitre.org/tactics/TA0008/)
- [MITRE ATT&CK: Persistence (TA0003)](https://attack.mitre.org/tactics/TA0003/)
- [GTFOBins](https://gtfobins.github.io/) — Unix 提權技術參考
- [LOLBAS](https://lolbas-project.github.io/) — Windows LOLBins 參考

## 📝 更新日誌

### v2.0.0 (2026-03-17)
- ✅ 實作 Windows 權限提升（Unquoted Paths/AIE/可寫服務/排程工作/Token 權限）
- ✅ 實作 Windows 持久化（登錄機碼 Run Keys/啟動資料夾/排程工作/服務）
- ✅ 統一 `detectors/` 路徑（重新匯出，消除雙重實作衝突）
- ✅ 修正 `engines/__init__.py` 匯出（新增整合型引擎別名）
- ✅ 修正 `detector/postex_detector.py` 的 import 命名錯誤
- ✅ 更正 README 狀態（移除錯誤的「✅ 完成」標記）
- ✅ 更正架構圖（移除不存在的 `linux_engine.py`、`windows_engine.py`、`credential_harvester.py`）

### v1.3.0 (2026-01-20)
- ✅ 移除 `postex_manager.py`（廢棄）
- ✅ 完善 Linux 權限提升、橫向移動、持久化引擎

### v1.2.0 (2025-12-17)
- ✅ PostExDetector 架構完成
- ✅ 基本檢測引擎實現

---

**維護者**: AIVA Team | **授權**: MIT License | **風險等級**: ⚠️ 高（需授權使用）
