# function_forensic - 數位鑑識模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，需人工操作 | **語言**: Python

## 🎯 模組概述

數位鑑識模組提供案件管理、證據採集、磁碟映像分析、記憶體傾印分析與時間線生成之工作流程框架。

> ⚠️ 本模組為高度人工介入之數位鑑識工作流框架，自動化能力有限。多數操作主要提供介面以供紀錄證據之 MD5/SHA256 雜湊值與調查時間線。

### 功能清單

| 功能 | 說明 |
|------|------|
| 案件管理 | 建立案件、管理調查人員與關聯證據 |
| 證據採集 | 建立磁碟/記憶體/檔案/日誌紀錄，自動計算雜湊驗證以確保完整性 |
| 磁碟映像分析 | 磁碟映像結構紀錄框架 |
| 記憶體傾印分析 | 記憶體分析流程紀錄框架 |
| 時間線生成 | 生成事件時間序列 |

## 📐 架構設計

```
function_forensic/
├── __init__.py    # 模組入口匯出
├── manager.py     # 鑑識流程管理器 (ForensicManager)
└── models.py      # 案件與證據資料模型 (CaseInfo, EvidenceItem 等)
```

## 🚀 執行方式

### 作為 Python 模組匯入

可透過 `ForensicManager` 來建立案件、登錄證據並確保證據完整性：

```python
from services.features.function_forensic import ForensicManager

manager = ForensicManager()

# 建立案件
case = manager.create_case(
    case_name="Investigation-001",
    investigator="analyst",
    description="Incident response"
)

# 採集證據 (將會針對來源檔案計算 Hash 並登錄)
evidence = await manager.acquire_evidence(
    case_id=case.id,
    source_path="/dev/sda",
    evidence_type="DISK",
    acquired_by="analyst"
)

# 標記分析任務
await manager.analyze_disk_image(evidence.id, deep_scan=True)
await manager.generate_timeline(evidence.id)
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `ForensicManager.create_case(...)` | 建立鑑識案件 |
| `ForensicManager.acquire_evidence(...)` | 登錄並採集證據檔案，自動計算 MD5/SHA256 |
| `ForensicManager.analyze_disk_image(...)` | 標記並開始磁碟映像分析 |
| `ForensicManager.analyze_memory_dump(...)` | 標記並開始記憶體傾印分析 |
| `ForensicManager.generate_timeline(...)` | 生成案件的所有事件時間線報告 |

## 🔒 證據類型

- `DISK`: 磁碟映像
- `MEMORY`: 記憶體傾印
- `FILE`: 個別檔案
- `LOG`: 系統/應用程式日誌檔案

## 注意事項

- 本模組作為鑑識工作流的骨幹，並未內建 Volatility 或 The Sleuth Kit 等二進位分析引擎，主要仰賴外部整合或人工操作。
- 採集的證據一旦登錄，雜湊值不得竄改以維持法庭證據能力 (Chain of Custody)。
- 無直接的 CLI 入口，不適用於一般黑盒滲透測試流程。
