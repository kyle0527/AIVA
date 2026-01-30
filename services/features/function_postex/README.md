# function_postex - 後滲透測試模組

> **版本**: v1.3.0 | **狀態**: ✅ 完成 | **語言**: Python | **更新**: 2026-01-20

## 🎯 模組概述

後滲透階段（Post-Exploitation）的安全檢測與分析模組，專注於已獲得初始訪問後的橫向移動、權限提升和持久化檢測。

### 核心功能

✅ **權限提升檢測** - 檢測 SUID/SUDO 誤配置、內核漏洞  
✅ **橫向移動分析** - 網絡掃描、內網服務發現、憑證收集  
✅ **持久化檢測** - 啟動項、計劃任務、後門檢測  
✅ **敏感數據收集** - 配置文件、歷史記錄、密鑰提取  
✅ **防禦規避** - AV/EDR 繞過建議、日誌清理檢測  

## 📐 架構設計

```
function_postex/
├── detector/
│   ├── postex_detector.py        # 核心檢測引擎
│   ├── privilege_escalation.py   # 權限提升檢測
│   ├── lateral_movement.py       # 橫向移動分析
│   └── persistence.py            # 持久化檢測
├── engines/
│   ├── linux_engine.py          # Linux 後滲透引擎
│   ├── windows_engine.py        # Windows 後滲透引擎
│   └── credential_harvester.py  # 憑證收集引擎
├── config/
│   └── postex_config.py         # 配置管理
├── postex_manager.py            # 高級管理接口（已廢棄）
└── README.md
```

## 🚀 快速開始

### 基本使用

```python
from services.features.function_postex.detector.postex_detector import PostExDetector

# 創建檢測器
detector = PostExDetector()

# 權限提升檢測（安全模式）
findings = detector.analyze(
    test_type="privilege_escalation",
    target="localhost",
    safe_mode=True  # 僅檢測，不執行
)

print(f"發現 {len(findings)} 個潛在權限提升向量")
```

### CLI 調用

```bash
# external_executor 調用
python aiva_external_executor.py \
    --lang python \
    --module postex \
    --test-type privilege_escalation \
    --safe-mode
```

## 🔧 功能詳解

### 1. 權限提升檢測

檢測可用於權限提升的系統誤配置：

**Linux**:
- SUID/SGID 可執行文件
- Sudo 誤配置 (`sudo -l`)
- 可寫入的 PATH 目錄
- 內核版本漏洞檢測
- Cron Job 誤配置
- Docker Socket 訪問

**Windows**:
- 未加引號的服務路徑
- 可修改的服務二進制
- AlwaysInstallElevated 註冊表鍵
- 計劃任務誤配置
- Token 劫持機會

```python
# 權限提升檢測
findings = detector.analyze(
    test_type="privilege_escalation",
    target="localhost",
    options={
        "check_suid": True,
        "check_sudo": True,
        "check_kernel": True,
        "safe_mode": True
    }
)

for finding in findings:
    print(f"{finding.severity}: {finding.title}")
    print(f"  向量: {finding.description}")
    print(f"  建議: {finding.recommendation}")
```

### 2. 橫向移動分析

分析可用於橫向移動的網絡服務和憑證：

**網絡掃描**:
- 內網主機發現
- 開放服務識別
- SMB/RDP/SSH 服務檢測
- 弱認證服務發現

**憑證收集**:
- SSH 私鑰提取
- 瀏覽器保存的密碼
- 配置文件中的憑證
- Kerberos 票據

```python
# 橫向移動分析
findings = detector.analyze(
    test_type="lateral_movement",
    target="192.168.1.0/24",
    options={
        "network_scan": True,
        "credential_harvest": True,
        "safe_mode": True
    }
)
```

### 3. 持久化檢測

檢測可用於建立持久化的系統位置：

**Linux**:
- RC 腳本和系統服務
- Cron Jobs
- SSH 授權密鑰
- LD_PRELOAD 劫持
- Bashrc/Profile 修改

**Windows**:
- 啟動項和註冊表運行鍵
- 計劃任務
- WMI 事件訂閱
- DLL 劫持機會
- 服務創建

```python
# 持久化檢測
findings = detector.analyze(
    test_type="persistence",
    target="localhost",
    options={
        "check_startup": True,
        "check_scheduled_tasks": True,
        "check_services": True,
        "safe_mode": True
    }
)
```

## 📊 輸出格式

```json
{
  "success": true,
  "test_type": "privilege_escalation",
  "target": "localhost",
  "safe_mode": true,
  "findings": [
    {
      "id": "POSTEX-001",
      "severity": "high",
      "title": "SUID Binary: /usr/bin/find",
      "description": "find 命令具有 SUID 位，可用於權限提升",
      "evidence": {
        "path": "/usr/bin/find",
        "permissions": "-rwsr-xr-x",
        "owner": "root"
      },
      "recommendation": "移除 SUID 位或限制訪問",
      "exploit_commands": [
        "find . -exec /bin/sh -p \\; -quit"
      ]
    }
  ],
  "summary": {
    "total_checks": 25,
    "vulnerabilities_found": 3,
    "high_severity": 1,
    "medium_severity": 2,
    "execution_time": "5.2s"
  }
}
```

## 🔒 安全模式

**⚠️ 重要**: 後滲透模組默認運行在**安全模式**下：

- ✅ **僅檢測**: 不執行任何破壞性操作
- ✅ **不修改系統**: 不創建文件、不修改配置
- ✅ **需授權**: 非安全模式需要明確的授權令牌

```python
# 安全檢測模式（默認）
findings = detector.analyze(
    test_type="privilege_escalation",
    safe_mode=True  # 強制啟用
)

# 真實測試模式（需授權）
findings = detector.analyze(
    test_type="privilege_escalation",
    safe_mode=False,
    auth_token="YOUR_AUTHORIZED_TOKEN"  # 必須提供
)
```

## 🎯 適用場景

✅ **滲透測試** - 後滲透階段的系統評估  
✅ **Red Team** - 橫向移動和權限提升路徑發現  
✅ **安全審計** - 系統配置安全性檢查  
✅ **合規檢測** - 特權訪問控制驗證  

**❌ 不適用於**:
- Bug Bounty（大多數平台禁止後滲透測試）
- 未授權系統測試
- 生產環境自動化掃描

## 📚 檢測規則庫

模組包含 **150+ 檢測規則**，涵蓋：

| 類別 | Linux 規則 | Windows 規則 |
|------|-----------|-------------|
| 權限提升 | 45 | 38 |
| 橫向移動 | 22 | 18 |
| 持久化 | 28 | 31 |
| 數據收集 | 15 | 12 |

## 🔗 相關資源

- [MITRE ATT&CK: Privilege Escalation](https://attack.mitre.org/tactics/TA0004/)
- [MITRE ATT&CK: Lateral Movement](https://attack.mitre.org/tactics/TA0008/)
- [MITRE ATT&CK: Persistence](https://attack.mitre.org/tactics/TA0003/)
- [GTFOBins](https://gtfobins.github.io/) - Unix 提權技術
- [LOLBAS](https://lolbas-project.github.io/) - Windows 提權技術

## 📝 更新日誌

### v1.3.0 (2026-01-20)
- ✅ 移除 postex_manager.py 統一包裝層
- ✅ 完善 privilege_escalation 檢測邏輯
- ✅ 完善 lateral_movement 分析引擎
- ✅ 完善 persistence 檢測規則
- ✅ 添加 150+ 新檢測規則
- ✅ 創建完整 README 文檔

### v1.2.0 (2025-12-17)
- ✅ PostExDetector 架構完成
- ✅ 基本檢測引擎實現

---

**維護者**: AIVA Team | **授權**: MIT License | **風險等級**: ⚠️ 高（需授權使用）
